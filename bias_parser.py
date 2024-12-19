from dataclasses import dataclass, field

@dataclass
class MoreConfig:
    ranks_per_gpu:int = field(
        default=16,
        metadata={"help": "Ranks per GPU"},
    )
    full_finetune:bool = field(
        default=False,
        metadata={"help": "全参数微调or部分微调"},
    )
    target_module:str = field(
        default= "[]",
        metadata={"help": "微调哪个部分"},
    )
    def __post_init__(self):
        self.target_module = self.parse_list(self.target_module)

    @staticmethod
    def parse_list(arg_value):
        if arg_value.startswith('[') and arg_value.endswith(']'):
            return arg_value.strip("[]").split(",")
        raise ValueError("Input should be formatted as a list: [item1,item2,...]")