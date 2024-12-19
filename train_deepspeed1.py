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
import deepspeed
from rich import print


PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def save_custom_model(model, tokenizer, model_path):
    torch.save(model, f"{model_path}.pth")
    tokenizer.save_pretrained(model_path)

class CustomLinearLayer(nn.Module):
    def __init__(self, original_linear, name, ranks_per_gpu=None):
        super(CustomLinearLayer, self).__init__()
        self.name = name
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # 获取原始权重并转换为 float32 以进行 SVD
        self.register_buffer('W', original_linear.weight.data.clone().detach())
        
        rank = ranks_per_gpu
        self.A = nn.Parameter(torch.zeros(rank, self.in_features)) 
        self.B = nn.Parameter(torch.randn(self.out_features, rank))
        
        for param in self.parameters():
            param.requires_grad = False

        # 设置 A 和 B 为可训练
        self.A.requires_grad = True
        self.B.requires_grad = True

        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.clone().detach())
        else:
            self.bias = None
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # 计算低秩偏差矩
        x_avg = x.mean(dim=0)
        result = torch.matmul(x_avg.t(), x_avg)
        hidden = nn.functional.linear(result, self.A)
        hidden = self.relu(hidden)
        bias_matrix = nn.functional.linear(hidden, self.B)
        # 将偏差矩阵加到原始权重矩阵上
        weight_with_bias = self.W + bias_matrix.t()
        
        # 计算输出
        return nn.functional.linear(x, weight_with_bias, self.bias)

    
    def __repr__(self):
        return (f"CustomLinearLayer(name={self.name}, "
                f"in_features={self.in_features}, out_features={self.out_features})")

def get_parent_module(model, module_name):
    module_name_parts = module_name.split('.')
    parent = model
    for part in module_name_parts[:-1]:
        parent = getattr(parent, part)
    return parent

def replace_with_custom_layer(model, target_modules, ranks_per_gpu=None):
    for name, module in model.named_modules():
        for target_name in target_modules:
            if target_name in name and isinstance(module, nn.Linear):
                parent_module = get_parent_module(model, name)
                setattr(parent_module, name.split('.')[-1], CustomLinearLayer(module, name, ranks_per_gpu))
                break


def main(args):
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed(verbose=True)
    
    ###########
    # model
    ###########
    
    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if args.bf16==True:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    replace_with_custom_layer(model, target_modules, ranks_per_gpu=args.ranks_per_gpu)
    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
    model = model.to(device)
    #########
    # Dataset
    #########

    # 加载数据集
    ds = load_dataset(args.dataset_name,split="train")
    
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
        
    # 应用预处理函数
    tokenized_datasets = filtered_ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    per_device_train_batch_size = args.batch_size
    num_devices = args.device_num
    total_steps = args.num_epochs * len(tokenized_datasets)//args.accumulation_steps//(per_device_train_batch_size*num_devices)
    warmup_steps = int(0.03 * total_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        model_parameters= filter(lambda p: p.requires_grad, model.parameters()),
        training_data=tokenized_datasets,
        dist_init_required=True,
        args=args,
    )
    
    loss_list = []
    current_step = 0
    os.makedirs(args.output_path, exist_ok=True)
    
    total_steps = args.num_epochs * len(training_dataloader)//args.accumulation_steps
    
    for epoch in range(args.num_epochs):
        for i,batch in enumerate(training_dataloader):
            current_step+=1
            input_ids = batch['input_ids'].to(device=model_engine.device)
            attention_mask = batch['attention_mask'].to(device=model_engine.device)
            labels = batch['labels'].to(device=model_engine.device)
            
            batch_mean = input_ids.float().mean().item()
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            
            loss_list.append(loss.item())
            current_lr = lr_scheduler.get_last_lr()[0]
            
            if args.local_rank==0:
                print(f"Batch Mean: {batch_mean:.6f}, Learning Rate: {current_lr:.8f}")
                with open(f'{args.output_path}/lr.txt','a') as f:
                    f.write(f"Batch Mean: {batch_mean:.6f}, Learning Rate: {current_lr:.8f}\n")

                with open(f'{args.output_path}/loss.txt','a') as file:
                    file.write(f'Step:{current_step} Loss:{loss.item()}\n')

            if current_step % 10 == 0 and args.local_rank == 0:
                    print(f"Step {current_step}/{total_steps} completed, remaining: {total_steps - current_step} steps.")
                    print(f"GPU {args.local_rank} processing step {current_step}, Loss: {loss_list[-1]}")
        
            if current_step % 500 == 0 and args.local_rank == 0 :
                model_path = os.path.join(args.output_path, f"saved_model_step_{current_step}")
                ensure_dir(os.path.dirname(model_path))
                save_custom_model(model.module, tokenizer, model_path)
                print(f"Model saved at step {current_step}")
        
        if args.local_rank == 0:
            print(f"Epoch {epoch + 1} completed.")

            model_path = os.path.join(args.output_path, f"saved_model_step_{current_step}")
            ensure_dir(os.path.dirname(model_path))
            model.module.save_pretrained(model_path)
            print(f"Model saved at step {current_step}")
        
        # 保存损失列表
    if args.local_rank  == 0:
        loss_list_path = os.path.join(args.output_path, "loss_list.pkl")
        with open(loss_list_path, 'wb') as f:
            pickle.dump(loss_list, f)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help='Model Path')
    parser.add_argument('--output_path', type=str, default='./output', help='Output Path')
    parser.add_argument('--dataset_name', type=str, default='meta-math/MetaMathQA', help='Output Path')
    parser.add_argument('--training_mode', type=str, default='bias_tuning', help='Training mode')
    parser.add_argument('--device_num', type=int, default=4, help='Ranks per GPU')
    parser.add_argument('--ranks_per_gpu', type=int, default=16, help='Ranks per GPU')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--bf16',type=bool,default=False,help='model dtype')
    parser.add_argument('--samedata', type=bool, default=False, help='Same or different data on GPUs')
    parser.add_argument('--local_rank', type=int, default=-1,
                help='local rank passed from distributed launcher')
    
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    print(args)
    main(args)