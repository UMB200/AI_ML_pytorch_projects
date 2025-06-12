import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import os
import zipfile
from pathlib import Path
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

def training_loop_step(model: torch.nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        loss_function: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        accuracy_fn,
                        device: torch.device = device):

  """ Performs a training with model trying to learn on data_loader"""
  ### Training
  training_loss, training_accuracy = 0, 0
  # Set model into training mode
  model.to(device)

  for batch, (X, y) in enumerate(data_loader):
      # Put data on target device
      X, y = X.to(device), y.to(device)
      # 1. Forward pass
      y_prediction = model(X)

      # 2. Calculate loss & accuracy per batch
      loss_value = loss_function(y_prediction, y)
      training_loss += loss_value # accumulate training loss
      training_accuracy += accuracy_fn(y_true = y,
                                        y_pred = y_prediction.argmax(dim=1)) # go from logits -> prediction labels
      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss_value.backward()

      # 5. Optimizer step
      optimizer.step()

  # Calculate the testing loss & accuracy by dividing of testing accuract by the length of test dataloader
  training_loss /= len(data_loader)
  training_accuracy /=len(data_loader)

  print(f"Training loss: {training_loss:.4f} |Training accuracy: {training_accuracy:.2f}%")
### Testing
def testing_loop_step(model: torch.nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      loss_function: torch.nn.Module,
                      accuracy_fn,
                      device: torch.device = device):
  testing_loss, testing_accuracy = 0, 0
  model.eval()
  model.to(device)
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      test_prediction = model(X)

      # 2. Calculate loss (accumulatively) & accuracy
      testing_loss += loss_function(test_prediction, y)
      testing_accuracy += accuracy_fn(y_true=y, y_pred=test_prediction.argmax(dim=1))

    # Calculate the test loss & accuracy average by dividing total testing loss and test accuracy by length of dataloader
    testing_loss /= len(data_loader)
    testing_accuracy /=len(data_loader)

    print(f"Testing loss: {testing_loss:.4f} | Testing accuracy: {testing_accuracy:.2f}%")
