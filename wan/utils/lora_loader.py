import torch
from safetensors import safe_open

class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype
    
    
    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
        return lora_name_dict


    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        updated_num = 0
        lora_name_dict = self.get_name_dict(state_dict_lora)
        for name, module in model.named_modules():
            if name in lora_name_dict:
                weight_up = state_dict_lora[lora_name_dict[name][0]].to(device=self.device, dtype=self.torch_dtype)
                weight_down = state_dict_lora[lora_name_dict[name][1]].to(device=self.device, dtype=self.torch_dtype)
                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)
                state_dict = module.state_dict()
                state_dict["weight"] = state_dict["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                module.load_state_dict(state_dict)
                updated_num += 1
        print(f"{updated_num} tensors are updated by LoRA.")

def load_state_dict_from_safetensors(file_path: str, device="cpu", torch_dtype=None)-> dict:
    state_dict = {}
    with safe_open(file_path, framework="pt", device=device) as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if torch_dtype is not None:
                tensor = tensor.to(torch_dtype)
            state_dict[key] = tensor
    return state_dict

def load_and_merge_lora_weight_from_safetensors(
    model: torch.nn.Module,
    lora_weight_path:str,
    device: str="cpu",
    torch_dtype = None,
    alpha: float=1.0,
    hotload: bool=False
) -> torch.nn.Module:
    lora_state_dict = load_state_dict_from_safetensors(lora_weight_path, torch_dtype=torch_dtype, device=device)

    if hotload:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                lora_a_name = f"{name}.lora_A.default.weight"
                lora_b_name = f"{name}.lora_B.default.weight"

                if lora_a_name in lora_state_dict and lora_b_name in lora_state_dict:
                    module.lora_A.weight.append(lora_state_dict[lora_a_name] * alpha)
                    module.lora_B.weight.append(lora_state_dict[lora_b_name])
    else:
        loader = GeneralLoRALoader(device=device, torch_dtype=torch_dtype)
        loader.load(model, lora_state_dict, alpha)

    return model