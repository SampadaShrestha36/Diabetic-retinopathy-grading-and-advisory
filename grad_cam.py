"""Grad-CAM explanations for the ResNet-50 diabetic retinopathy classifier."""

from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD
from preprocessing import preprocess_image


class GradCAM:
    """Computes a class-specific Grad-CAM map from a convolutional layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output):
        self.activations = output

    def _save_gradients(self, _module, _grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image_tensor, class_index=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        logits[:, class_index].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True).relu()
        cam = torch.nn.functional.interpolate(
            cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        maximum = cam.max()
        if maximum > 0:
            cam /= maximum
        return cam, logits[0].detach()


def _denormalize(image_tensor):
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image * std + mean, 0, 1)


def _add_panel(canvas, panel, title, x, y, panel_size):
    """Places a labeled, consistently sized panel on the explanation board."""
    panel = cv2.resize(panel, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC)
    canvas[y:y + panel_size, x:x + panel_size] = panel
    cv2.rectangle(canvas, (x, y), (x + panel_size - 1, y + panel_size - 1),
                  (8, 8, 8), 6)
    cv2.rectangle(canvas, (x + 4, y + 4),
                  (x + panel_size - 5, y + panel_size - 5), (245, 245, 245), 2)
    cv2.rectangle(canvas, (x + 6, y + 6), (x + panel_size - 7, y + 48), (15, 15, 15), -1)
    cv2.line(canvas, (x + 6, y + 48), (x + panel_size - 7, y + 48),
             (245, 245, 245), 2)
    cv2.putText(canvas, title, (x + 14, y + 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (255, 255, 255), 2, cv2.LINE_AA)


def _make_visualization(image, cam, predicted_class, confidence, true_class):
    panel_size = 512
    gutter = 16
    header_height = 112
    board_width = panel_size * 2 + gutter * 3
    board_height = header_height + panel_size * 2 + gutter * 3

    image_uint8 = (image * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_uint8, 0.52, heatmap, 0.48, 0)

    # Keep only the strongest activations in this view so the evidence is easier to inspect.
    focus_mask = (cam >= 0.55).astype(np.float32)[..., None]
    focus = np.clip(image * (0.28 + 0.72 * focus_mask) + heatmap * 0.38 * focus_mask, 0, 1)
    focus = (focus * 255).astype(np.uint8)

    board = np.full((board_height, board_width, 3), 28, dtype=np.uint8)
    cv2.putText(board, "Grad-CAM explanation", (gutter, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.95, (255, 255, 255), 2, cv2.LINE_AA)
    summary = (
        f"Model evidence for grade {predicted_class}  |  "
        f"Confidence: {confidence:.1%}  |  True grade: {true_class}"
    )
    cv2.putText(board, summary, (gutter, 76), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (205, 220, 230), 1, cv2.LINE_AA)

    top = header_height + gutter
    left = gutter
    _add_panel(board, image_uint8, "1  Input fundus image", left, top, panel_size)
    _add_panel(board, heatmap, "2  Attention strength: blue to red", left + panel_size + gutter,
               top, panel_size)
    _add_panel(board, overlay, "3  Evidence overlaid on the image", left,
               top + panel_size + gutter, panel_size)
    _add_panel(board, focus, "4  Strongest evidence (>= 0.55)", left + panel_size + gutter,
               top + panel_size + gutter, panel_size)

    cv2.putText(board, "red = stronger contribution", (left + panel_size + gutter + 14,
                top + panel_size + gutter + panel_size - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(board, cv2.COLOR_RGB2BGR)


def generate_gradcam(model, dataframe, transform, device, output_dir, limit=None):
    """Save original, heatmap, and overlay panels for rows in a label dataframe."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    explainer = GradCAM(model, model.layer4[-1])
    generated = 0

    try:
        for row_index, row in dataframe.iterrows():
            if limit is not None and generated >= limit:
                break
            image = preprocess_image(row["filepath"], size=224)
            if image is None:
                print(f"Skipping Grad-CAM for low-quality/unreadable image: {row['filepath']}")
                continue

            image_tensor = transform(image).unsqueeze(0).to(device)
            with torch.enable_grad():
                cam, logits = explainer(image_tensor)
            probabilities = torch.softmax(logits, dim=0)
            predicted_class = int(probabilities.argmax().item())
            confidence = float(probabilities[predicted_class].item())
            visualization = _make_visualization(
                _denormalize(image_tensor[0]), cam, predicted_class, confidence, int(row["label"])
            )

            source_name = Path(str(row["filepath"])).stem
            output_file = output_path / f"{row_index:05d}_{source_name}_gradcam.jpg"
            cv2.imwrite(str(output_file), visualization)
            generated += 1
            print(f"Grad-CAM [{generated}] saved: {output_file}")
    finally:
        explainer.remove_hooks()

    return generated
