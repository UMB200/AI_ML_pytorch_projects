import torch
import torchvision
import argparse
import model_builder

# Create a parser
parser = argparse.ArgumentParser()

# Get an image path
parser.add_argument("--image",
                    help="target image filepath to predict on")
# Get model path
parser.add_argument("--model_path",
                    default="models/05_going_modular_script_mode_tinyvgg_model.pth",
                    type=str,
                    help="target model to use for prediction filepath")

args = parser.parse_args()

# Setup class names
class_names = ["pizza", "steak", "sushi"]

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# image path
IMG_PATH = args.image
print(f"[INFO] Predicting on {IMG_PATH}")

# Loading in the model
def load_model(model_path=args.model_path):
  model = model_builder.Model_Builder_TinyVGG(input_shape=3,
                                             hidden_units=128,
                                             output_shape=3).to(device)
  print(f"[INFO] Loading in model from: {model_path}")
  model.load_state_dict(torch.load(model_path))
  return model

# Load in model and predict image
def predict_image(image_path=IMG_PATH,
                      model_path=args.model_path):
  # Load in image
  image = torchvision.io.read_image(str(IMG_PATH)).type(torch.float32)
  model = load_model(model_path)

  # Process image to be the same as in model
  image = image / 255.
  transform = torchvision.transforms.Resize(size=(64, 64))
  image = transform(image)

  # Predict the image
  model.eval()
  with torch.inference_mode():
    image = image.to(device)
    pred_logits = model(image.unsqueeze(dim=0))
    pred_probs = torch.softmax(pred_logits, dim=1)
    pred_labels = torch.argmax(pred_probs, dim=1)
    pred_labels_class = class_names[pred_labels]
  print(f"[INFO] Pred class: {pred_labels_class}, Pred prob: {pred_probs.max():.3f}")

if __name__ == "__main__":
  predict_image()
