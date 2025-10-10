import torch
def set_rand_seed(seed: int=42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
