"""Day 5: Demo notebook for teacher presentation.

Loads up to 3 condition bundles (A/B/C) and runs them side-by-side on a small set
of out-of-distribution Google images. Shows prediction probability bar chart +
Grad-CAM heatmap per image per condition.

Usage:
- Put OOD images in a folder (e.g. `demo_images/`)
- Set BUNDLE_PATHS to the 3 fine-tune bundles
- Run all cells; screenshots feed the teacher slides
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augmentation import build_eval_transform, build_tta_transforms
from src.model import ClassifierModel
from src.config import CLASS_NAMES

BUNDLE_PATHS = {
    "A_scratch": "outputs/finetune/demo_bundle_A_scratch_fold0.pth",
    "B_simclr": "outputs/finetune/demo_bundle_B_simclr_fold0.pth",
    "C_imagenet": "outputs/finetune/demo_bundle_C_imagenet_fold0.pth",
}
DEMO_IMAGE_DIR = Path("demo_images")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_bundle(path: str) -> ClassifierModel:
    state = torch.load(path, map_location=device)
    pretrained = state["condition"] == "C_imagenet"
    model = ClassifierModel(pretrained_backbone=pretrained)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def predict(img_path: Path, model: ClassifierModel, use_tta: bool = True) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    if use_tta:
        tfs = build_tta_transforms()
        views = [tf(img).unsqueeze(0).to(device) for tf in tfs]
        probs = torch.stack([F.softmax(model(v), dim=1) for v in views]).mean(0)
    else:
        tf = build_eval_transform()
        probs = F.softmax(model(tf(img).unsqueeze(0).to(device)), dim=1)
    return probs.squeeze().detach().cpu().numpy()


def gradcam_heatmap(img_path: Path, model: ClassifierModel, target_class: int):
    """Returns (H, W, 3) RGB overlay using last conv block as target layer."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    tf = build_eval_transform()
    img = Image.open(img_path).convert("RGB")
    img_resized = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    input_tensor = tf(img).unsqueeze(0).to(device)

    target_layer = [model.backbone.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layer)
    grayscale = cam(input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(target_class)])[0]
    return show_cam_on_image(img_resized, grayscale, use_rgb=True)


def show_image_across_conditions(img_path: Path, bundles: dict):
    n_cond = len(bundles)
    fig, axes = plt.subplots(n_cond, 3, figsize=(15, 4 * n_cond))
    if n_cond == 1:
        axes = axes.reshape(1, -1)

    img = Image.open(img_path).convert("RGB")
    for row, (cond, model) in enumerate(bundles.items()):
        probs = predict(img_path, model)
        top1 = int(np.argmax(probs))
        cam_img = gradcam_heatmap(img_path, model, top1)

        axes[row, 0].imshow(img); axes[row, 0].axis("off")
        axes[row, 0].set_title(f"{cond}\nPred: {CLASS_NAMES[top1]} ({probs[top1]:.1%})")

        axes[row, 1].barh(range(10), probs, color="steelblue")
        axes[row, 1].set_yticks(range(10))
        axes[row, 1].set_yticklabels(CLASS_NAMES, fontsize=8)
        axes[row, 1].set_xlim(0, 1); axes[row, 1].set_xlabel("Probability")
        axes[row, 1].invert_yaxis()

        axes[row, 2].imshow(cam_img); axes[row, 2].axis("off")
        axes[row, 2].set_title("Grad-CAM")

    plt.suptitle(img_path.name, fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    bundles = {}
    for cond, path in BUNDLE_PATHS.items():
        if Path(path).exists():
            bundles[cond] = load_bundle(path)
            print(f"Loaded: {cond} from {path}")
        else:
            print(f"Skip: {cond} ({path} not found)")

    if not bundles:
        raise SystemExit("No bundles loaded. Train models first.")

    image_files = sorted(DEMO_IMAGE_DIR.glob("*.jpg")) + sorted(DEMO_IMAGE_DIR.glob("*.png"))
    if not image_files:
        raise SystemExit(f"No images in {DEMO_IMAGE_DIR}/")

    for img_path in image_files:
        show_image_across_conditions(img_path, bundles)
