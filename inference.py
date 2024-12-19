import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import math
from tqdm import tqdm
import pickle
from peft import get_peft_model, LoraConfig
from transformers import get_cosine_schedule_with_warmup
import argparse

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from peft import LoraConfig, get_peft_model
from peft import PeftModel

from bias_model import (
    replace_with_custom_layer,print_trainable_parameters,load_model
)
from bias_parser import MoreConfig

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Configuration for Model Fine-tuning")
    parser.add_argument("--finetune_model_path", type=str,default="/scratch/limuyao/checkpoints/Qwen2.5_0.5B_Instruct-bias_finetune-v_proj-12_18_01-c4-e1-b16-a4/checkpoint-928")
    parser.add_argument("--base_model_path", type=str,default="/scratch/limuyao/checkpoints/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", type=int,default=5)
    parser.add_argument("--full_finetune", type=bool, default=False)
    parser.add_argument("--ranks_per_gpu", type=int, default=16, help="Ranks per GPU")
    parser.add_argument("--target_module", type=str, default="[k_proj]", help="Specify which parts of the model to fine-tune")
    args = parser.parse_args()
    config = MoreConfig(ranks_per_gpu=args.ranks_per_gpu, full_finetune=args.full_finetune, target_module=args.target_module)
    print(config)
    
    finetune_model_path = args.finetune_model_path
    base_model_path = args.base_model_path
    target_modules = config.target_module
    device = args.device
    
    # 加载 GSM8K 数据集
    gsm8k_ds = load_dataset("/scratch/limuyao/datas/gsm8k", 'main', split="test")

    # 加载 tokenizer和model
    from transformers import AutoModelForCausalLM,AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    if len(target_modules)>0 and target_modules[0]!='':
        print(target_modules)
        replace_with_custom_layer(model, target_modules, ranks_per_gpu=config.ranks_per_gpu)
        load_model(model,finetune_model_path)
        
    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
    print_trainable_parameters(model,record_path="show.json")

    # 推理配置
    model.eval().to(device)

    # 准确率统计
    correct = 0
    total = 0

    # 推理过程
    for item in tqdm(gsm8k_ds):
        # 准备问题和答案
        question = PROMPT.format(instruction=item['question'])
        true_answer = item['answer'].split('#### ')[-1].strip()

        # Tokenize 输入
        inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)

        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
        
        # 解码模型生成的输出
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # 对比模型生成的答案和真实答案
        if true_answer in generated_answer:  # 简单包含匹配
            correct += 1
        total += 1
        with open(f"{finetune_model_path.split('/')[-2]}.txt", 'a' ) as f:
            print(f"Question: {item['question']}",file=f)
            print(f"True Answer: {true_answer}",file=f)
            print(f"Generated Answer: {generated_answer}",file=f)
            print(f"Correct so far: {correct}/{total}",file=f)

    # 输出最终准确率
    accuracy = correct / total * 100
    print(f"Final Accuracy: {accuracy:.2f}%")
