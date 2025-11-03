"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""
import os
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """

  # Use ImageFolder to create dataset(s)
  train_dataset = datasets.ImageFolder(train_dir, transform=transform)
  test_dataset = datasets.ImageFolder(test_dir,transform=transform)

  # Get class names
  class_names = train_dataset.classes

  # Turn datasets to DataLoaders
  train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True,
                                pin_memory=True)
  test_dataloader = DataLoader(dataset=test_dataset,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               shuffle=False,
                               pin_memory=True)

  return train_dataloader, test_dataloader, class_names


#Split Data to 20 and 80% from Fodd101
def split_data(
    dataset:datasets,
    split_size:float=0.2,
    seed:int=42):

  first_length_part = int(split_size * len(dataset))
  remaining_length_part = len(dataset) - first_length_part

  print(f"Splitting dataset {len(dataset):,} images into 2 sets: 1 set is  {first_length_part:,} images, second one is {remaining_length_part:,} images")
  print(f"{first_length_part:,} equals to {int(split_size*100)}% of total data")
  print(f"{remaining_length_part:,} equals to {int(100-split_size*100)}% % of total data")

  dataset_A, dataset_B = torch.utils.data.random_split(
      dataset=dataset,
      lengths=[first_length_part, remaining_length_part],
      generator=torch.manual_seed(seed))
  return dataset_A, dataset_B
