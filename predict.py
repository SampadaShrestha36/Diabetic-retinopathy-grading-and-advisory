"""
Loads a previously trained model checkpoint (.pt file) so you can
run predictions without retraining from scratch.
"""

import torch
from model import build_model
from preprocessing import preprocess_image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(checkpoint_path="best_model_stage2.pt", num_classes=5):
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # important: switches off dropout/batchnorm training behavior
    return model


def predict_single_image(model, image_path):
    image = preprocess_image(image_path, size=224)
    if image is None:
        print("Image failed quality check")
        return None

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        output = model(tensor)
        predicted_grade = torch.argmax(output, dim=1).item()

    return predicted_grade


if __name__ == "__main__":
    model = load_trained_model("best_model_stage2.pt")

    grade = predict_single_image(model, "data/aptos2019/train_images/000c1434d8d7.png")
    print(f"Predicted DR grade: {grade}")