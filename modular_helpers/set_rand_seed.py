import torch
def set__rand_seed(seed: int=42):
  """ set random seed for toech operations
  Args:
        seed(int, optional): Random seed to set, default is 42
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
