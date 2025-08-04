#get helper functions from github
import os
import zipfile
from pathlib import Path
import sys

# Clone the repository
!git clone https://github.com/UMB200/pytorch_projects.git

# Create the new directory if it doesn't exist
if not os.path.exists('/content/going_modular'):
  !mkdir /content/going_modular

# Move all files from modular_helpers to the new directory
# Use a wildcard to select files, excluding directories if possible, though mv handles this gracefully
!mv /content/pytorch_projects/modular_helpers/* /content/going_modular/

# Remove the cloned repository
!rm -rf /content/pytorch_projects

# Add the path to the going_modular directory to sys.path so you can import from it
sys.path.insert(0, '/content/going_modular')
