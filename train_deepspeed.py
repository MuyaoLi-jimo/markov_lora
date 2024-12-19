from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import os
import pathlib
from typing import Optional
from datasets import load_dataset
from bias_parser import MoreConfig
from bias_model import (
    replace_with_custom_layer,print_trainable_parameters
)
TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)
from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"
    from rich.console import Console
    from rich.logging import RichHandler
from transformers import Trainer
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

class CustomDeepSpeedTrainer(Trainer):

    def custom_save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict() 
        torch.save(state_dict, os.path.join(output_dir,"model.bin"))
        
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
            
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
            self.custom_save(output_dir, state_dict=state_dict)
        else:
            self.custom_save(output_dir)
    

class DataCollator:
    def __init__(self,):
        pass

    def __call__(self, examples):
        keys = ['input_ids','attention_mask','labels']
        inputs = { key: [example[key] for example in examples] for key in keys}
        for key in keys:
            inputs[key] = torch.stack(inputs[key],dim=0)
        return inputs

if __name__ == "__main__":
    
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig,MoreConfig))
    sft_script_args, training_args, model_config,more_config = parser.parse_args_and_config()
    
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()
    
    #######
    # model
    #######
    
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path,)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        low_cpu_mem_usage=True,
        **model_kwargs,
    )
    target_modules = more_config.target_module #"q_proj","o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    
    if target_modules and target_modules[0]!="":
        replace_with_custom_layer(model, target_modules, ranks_per_gpu=more_config.ranks_per_gpu)
    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
    
    # 只微调部分参数
    if not more_config.full_finetune:
        for name, param in model.named_parameters():
            grad_flag = False
            for target_module in target_modules:
                if target_module in name:
                    grad_flag = True
            param.requires_grad = grad_flag
        
    print_trainable_parameters(model,record_path="show.json")
    
    ##################
    #  Dataset
    ##################
    
    ds = load_dataset(sft_script_args.dataset_name,split="train")
    # 定义筛选函数
    def filter_function(example):
        # # 排除类型为 "MATH" 的数据
        if "MATH" in example["type"]:
            return False
        return True

    # 对训练集进行筛选
    filtered_ds = ds.filter(filter_function)

    # 训练集的 token 化
    def preprocess(examples):
        # 拼接 query 和 response 后进行整体 tokenization
        concatenated_inputs = [
            PROMPT.format(instruction=q) + f"{a}\n{tokenizer.eos_token}"
            for q, a in zip(examples["query"], examples["response"]) 
            if q.strip() != "" and a.strip() != ""
        ]
        
        tokenized = tokenizer(
            concatenated_inputs,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        
        # 创建 labels，将输入部分标记为 IGNORE_INDEX
        labels = tokenized['input_ids'].clone()
        input_ids_lens = [len(tokenizer(PROMPT.format(instruction=q))['input_ids']) for q in examples["query"]]
        for label, source_len in zip(labels, input_ids_lens):
            label[:source_len] = -100  # 忽略输入部分
        pad_mask = tokenized['input_ids'] == tokenizer.pad_token_id

        first_pad_indices = pad_mask.int().argmax(dim=1)  # 获取每行第一个 pad_token 的位置

        # 对所有位置进行 mask，将第一个 pad_token 位置排除
        for i, first_pad_idx in enumerate(first_pad_indices):
            pad_mask[i, first_pad_idx] = False  # 排除第一个 pad_token 的位置

        # 将剩余的 pad_token 的 labels 设置为 -100
        labels[pad_mask] = -100

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }

    # 测试集的 token 化
    def tokenize_function_test(examples):
        # 仅对 query 进行编码
        inputs = tokenizer(examples["question"], truncation=True, max_length=512)
        # 将 response 保存下来，供后续评估使用
        outputs = tokenizer(examples["answer"], truncation=True, max_length=512)
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': outputs['input_ids']
        }

    # 应用预处理函数
    tokenized_datasets = filtered_ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.01)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]
    
    ##############
    #  Train
    ##############
    
    trainer = CustomDeepSpeedTrainer( 
        model=model,
        args=training_args,
        data_collator=DataCollator(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
    )
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    from contextlib import nullcontext
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )
    with save_context:
        trainer.save_model(training_args.output_dir)