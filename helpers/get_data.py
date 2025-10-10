"""
Downloads data from source and saves to a target directory.
"""

import os
import zipfile
from pathlib import Path
import requests
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Setup path to data folder
data_path = Path("data_path/")
img_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exists, download it and prepare it
if img_path.is_dir():
  print(f"{img_path} directory already exists")
else:
  print(f"Creating directory {img_path}")
  img_path.mkdir(parents=True, exist_ok=True)
# Download data
url_to_download = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
file_path = data_path / "pizza_steak_sushi.zip"
with open(file_path, "wb") as f:
  print(f"Downloading data")
  f.write(requests.get(url_to_download).content)
#Unzip data
with zipfile.ZipFile(file_path, "r") as zip_ref:
  print(f"Unzipping: {file_path}")
  zip_ref.extractall(img_path)

# Remove zip file
os.remove(file_path)
