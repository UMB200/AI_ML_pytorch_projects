import os
import zipfile
import requests
from pathlib import Path

def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    data_path = Path("data/")
    image_path = data_path / destination
    target_file = Path(source).name
    target_path = data_path / target_file
    
    if image_path.is_dir():
        print(f"{image_path} dir exists, moving on")
    else:
        print(f"No {image_path} dir is found, creating one")
        image_path.mkdir(parents=True, exist_ok=True)

        with open(target_path, "wb") as file:
            print(f"Downloading {target_file} from {source}")
            file.write(requests.get(source).content)
        with zipfile.ZipFile(target_path, "r") as zr:
            zr.extractall(image_path)
        if remove_source:
            os.remove(target_path)
    return image_path
