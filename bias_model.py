import torch.nn as nn
import torch
import os

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

def load_model(model: nn.Module, model_name_or_path: str):
    model_path = os.path.join(model_name_or_path, "model.bin")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    
    return model  # 返回加载后的模型

def print_trainable_parameters(model:torch.nn.Module,optimizer:torch.optim.Optimizer=None,record_path = None):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    model_shapes = []
    for name, parameter in model.named_parameters():
        if optimizer:
            optimizer_group_idx = None
            for idx,param_group in enumerate(optimizer.param_groups):
                for param in param_group["params"]:
                    if parameter is param:
                        optimizer_group_idx = idx
            model_shapes.append([parameter.requires_grad,name,parameter.shape,optimizer_group_idx])
        else:
            model_shapes.append([parameter.requires_grad,name,parameter.shape])
        all_param += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()
    import json
    if record_path:
        with open(record_path,mode="w",encoding="UTF-8") as f:
            json.dump(model_shapes, f, indent=4)
        
        with open(record_path.replace(".json","-scratch.txt"),mode="w",encoding="UTF-8") as f:
            print(optimizer, file=f)
            print(model, file=f)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    