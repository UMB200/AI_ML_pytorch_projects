def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
  """ Returns a dictionary containing the results of model predicting on data_loader

  Args:
    model (torch.nn.Module): a Python model capabale of making predictions on data_loader.
    data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
    loss_function (torch.nn.Module): The loss function of the model.
    accuracy_fn: Accuracy function to compare model's predictions to the truth labels
  Returns:
    (dict): results of model making predictions on data_loader.
"""
  loss_value, accuracy_value = 0, 0
  model.eval()
  model.to(device)
  with torch.inference_mode():
    for X, y in tqdm(data_loader):
      # Make data device agnostic
      X, y = X.to(device), y.to(device)
      # Make predictions
      y_predictions = model(X)

      # Accumulate loss and accuracy values per batch
      loss_value += loss_function(y_predictions, y)
      accuracy_value += accuracy_fn(y_true=y,
                                    y_pred = y_predictions.argmax(dim=1))
    # Scale loss and accuracy to find the average loss/acc per batch
    loss_value /= len(data_loader)
    accuracy_value /= len(data_loader)

  return {"model_name": model.__class__.__name__,
          "model_loss": loss_value.item(),
          "model_acc": accuracy_value}
