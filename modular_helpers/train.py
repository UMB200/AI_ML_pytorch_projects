"""
Trains Python model for image classification using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils
from torchvision import transforms
from timeit import default_timer as timer
import argparse

parser = argparse.ArgumentParser(description="Getting hyperparameters")

parser.add_argument("--num_epochs",
                    default=5,
                    type=int,
                    help="number of epochs to train for")
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="number of samples per batch")
parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help="number of hidden units in hidden layers")
parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help="learning rate to use for model")
parser.add_argument("--test_dir",
                    default="data_path/pizza_steak_sushi/test",
                    type=str,
                    help="directory of test data")
parser.add_argument("--train_dir",
                    default="data_path/pizza_steak_sushi/train",
                    type=str,
                    help="directory of train data")

args = parser.parse_args()

# SETUP HYPERPARAMETERS
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate


# SETUP DIRECTORIES
train_dir_scripted = "data_path/pizza_steak_sushi/train"
test_dir_scripted =  "data_path/pizza_steak_sushi/test"

# SETUP TARGET DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"

# CREATE TRANSFORMS
data_transform = transforms.Compose([transforms.Resize(size=(64, 64)),
                                     transforms.ToTensor()])

# CREATE DATALOADERS USING data_setup.py
train_dataloader_scripted, test_dataloader_scripted, class_names_scripted = data_setup.create_dataloaders(
    train_dir = train_dir_scripted,
    test_dir = test_dir_scripted,
    transform = data_transform,
    batch_size = BATCH_SIZE)

# CREATE MODEL USING model_builder.py
model_saved = model_builder.Model_Builder_TinyVGG(input_shape=3,
                                             hidden_units=HIDDEN_UNITS,
                                             output_shape=len(class_names_scripted)).to(device)

# SETUP LOSS_FN AND OPTIMIZER
loss_fn_cross_entropy = torch.nn.CrossEntropyLoss()
optimizer_adam = torch.optim.Adam(model_saved.parameters(), lr=LEARNING_RATE)

# START TRAINING USING engine.py
star_time = timer()
engine.train(model=model_saved,
             train_dataloader = train_dataloader_scripted,
             test_dataloader = test_dataloader_scripted,
             loss_fn = loss_fn_cross_entropy,
             optimizer = optimizer_adam,
             epochs = NUM_EPOCHS,
             device = device)
end_time = timer()
print(f"[INFO] Total training time: {end_time-star_time:.3f} seconds")

# SAVE MODEL USING utils.py
utils.save_model(model = model_saved,
                 target_dir = "models",
                 model_name = "05_going_modular_script_mode_tinyvgg_model.pth")
