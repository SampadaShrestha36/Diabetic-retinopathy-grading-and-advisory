"""
Gradio app: upload a fundus image, get a DR severity grade and a Grad-CAM
explanation of which regions drove the prediction.

Preprocessing mirrors preprocessing.py's preprocess_image exactly (this
app calls it directly rather than reimplementing any step): load -> RGB
convert -> circular_crop -> quality check (blur/darkness) -> CLAHE or Ben
Graham enhancement -> resize to 224x224.

IMPORTANT: set USE_BEN_GRAHAM below to whatever value you actually passed
to train.py's --ben_graham flag when training best_model_stage2.pt. If
the checkpoint was trained with Ben Graham enhancement but this app runs
CLAHE instead (or vice versa), the model sees systematically different
inputs at inference than it learned on, which can quietly hurt accuracy
with no error or warning.

(Monte Carlo Dropout uncertainty removed for now — your current model.py
has no dropout layers, so that part needed a modified architecture. Add
it back later once you're ready to revisit that.)
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import gradio as gr

from model import build_model
from dataset import get_eval_transforms, IMAGENET_MEAN, IMAGENET_STD
from preprocessing import preprocess_image
from grad_cam import GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "best_model_stage2.pt"

# Must match whichever flag value was used when this checkpoint was
# trained (train.py's --ben_graham). False = CLAHE (preprocess_image's
# default), True = Ben Graham local-average subtraction.
USE_BEN_GRAHAM = False

GRADE_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


def load_trained_model(checkpoint_path, num_classes=5):
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def denormalize(image_tensor):
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image * std + mean, 0, 1)


def make_gradcam_overlay(image_tensor, cam):
    image = denormalize(image_tensor[0])
    image_uint8 = (image * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_uint8, 0.55, heatmap, 0.45, 0)
    return overlay


# ---- Load once at startup, not per-request ----
model = load_trained_model(CHECKPOINT_PATH)
transform = get_eval_transforms()
explainer = GradCAM(model, model.layer4[-1])


def predict(image_path):
    if image_path is None:
        return None, "Please upload a fundus image.", None

    image = preprocess_image(image_path, size=224, use_ben_graham=USE_BEN_GRAHAM)
    if image is None:
        return None, "Image failed quality checks (unreadable or too low quality).", None

    image_tensor = transform(image).unsqueeze(0).to(device)

    cam, logits = explainer(image_tensor)
    probs = F.softmax(logits, dim=0).cpu().numpy()
    predicted_class = int(probs.argmax())
    confidence = float(probs[predicted_class])
    overlay = make_gradcam_overlay(image_tensor, cam)

    summary = (
        f"Predicted grade: {predicted_class} — {GRADE_NAMES[predicted_class]}\n"
        f"Confidence: {confidence:.1%}"
    )

    class_probs = {f"{i} — {GRADE_NAMES[i]}": float(probs[i]) for i in range(5)}

    return overlay, summary, class_probs


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Upload fundus image"),
    outputs=[
        gr.Image(label="Grad-CAM explanation"),
        gr.Textbox(label="Prediction", lines=3),
        gr.Label(label="Class probabilities", num_top_classes=5),
    ],
    title="Diabetic Retinopathy Grading",
    description=(
        "Upload a fundus image to get a DR severity grade (0-4) and a "
        "Grad-CAM heatmap showing which regions drove the prediction."
    ),
)

if __name__ == "__main__":
    demo.launch()