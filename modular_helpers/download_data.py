import os
import zipfile
from pathlib import Path

def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
  
