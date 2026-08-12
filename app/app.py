from pathlib import Path
from collections import Counter

import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


# ----------------------------
# 기본 설정
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "resnet18_font_cls_best_v2.pth"

IMAGE_SIZE = 224
PATCH_SIZE = 512
STRIDE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# patch 좌표 생성
# ----------------------------
def get_positions(length, patch_size, stride):
    if length < patch_size:
        return []

    positions = list(range(0, length - patch_size + 1, stride))

    last_pos = length - patch_size
    if positions and positions[-1] != last_pos:
        positions.append(last_pos)

    return positions


# ----------------------------
# text patch 판별
# ----------------------------
def is_text_patch(
    patch,
    min_mean_brightness=80,
    min_edge_ratio=0.008,
    min_component_area=100,
    min_component_count=3
):
    img = np.array(patch)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if np.mean(gray) < min_mean_brightness:
        return False

    bg_est = np.percentile(gray, 90)
    threshold = max(0, int(bg_est - 25))

    binary = (gray < threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = np.mean(edges > 0)

    if edge_ratio < min_edge_ratio:
        return False

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    large_components = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            large_components += 1

    if large_components < min_component_count:
        return False

    return True


# ----------------------------
# patch 추출
# ----------------------------
def extract_patches(image):
    width, height = image.size

    x_positions = get_positions(width, PATCH_SIZE, STRIDE)
    y_positions = get_positions(height, PATCH_SIZE, STRIDE)

    patches = []
    coords = []

    for y in y_positions:
        for x in x_positions:
            patch = image.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            patches.append(patch)
            coords.append((x, y))

    return patches, coords


# ----------------------------
# 모델 로드
# ----------------------------
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


model, class_names = load_model()


# ----------------------------
# 추론용 transform
# ----------------------------
infer_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ----------------------------
# patch 단위 추론
# ----------------------------
def predict_patch_probs(patch):
    x = infer_transform(patch).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]

    return probs, pred_idx, pred_name


# ----------------------------
# 전체 이미지 추론
# ----------------------------
def predict_font(image):
    if image is None:
        return "이미지를 업로드하세요.", None

    image = image.convert("RGB")
    width, height = image.size

    if width < PATCH_SIZE or height < PATCH_SIZE:
        return f"이미지가 너무 작습니다. 최소 {PATCH_SIZE}x{PATCH_SIZE} 이상이어야 합니다.", None

    patches, coords = extract_patches(image)

    valid_patches = []
    valid_coords = []

    for patch, coord in zip(patches, coords):
        if is_text_patch(patch):
            valid_patches.append(patch)
            valid_coords.append(coord)

    if len(valid_patches) == 0:
        return "텍스트가 있는 patch를 찾지 못했습니다.", None

    all_probs = []
    patch_preds = []

    for patch in valid_patches:
        probs, pred_idx, pred_name = predict_patch_probs(patch)
        all_probs.append(probs)
        patch_preds.append(pred_name)

    mean_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    final_idx = int(np.argmax(mean_probs))
    final_name = class_names[final_idx]
    final_conf = float(mean_probs[final_idx])

    top3_idx = np.argsort(mean_probs)[::-1][:3]
    top3_lines = []
    for rank, idx in enumerate(top3_idx, start=1):
        top3_lines.append(f"{rank}. {class_names[idx]}: {mean_probs[idx]:.4f}")

    count_text = Counter(patch_preds)
    count_lines = [f"{name}: {count}" for name, count in count_text.most_common()]

    result_text = (
        f"최종 예측 폰트: {final_name} ({final_conf:.4f})\n\n"
        f"전체 patch 수: {len(patches)}\n"
        f"사용된 text patch 수: {len(valid_patches)}\n\n"
        f"[상위 3개 평균 확률]\n" +
        "\n".join(top3_lines) +
        "\n\n[patch 예측 개수]\n" +
        "\n".join(count_lines)
    )

    preview = valid_patches[0] if len(valid_patches) > 0 else None
    return result_text, preview


# ----------------------------
# Gradio UI
# ----------------------------
demo = gr.Interface(
    fn=predict_font,
    inputs=gr.Image(type="pil", label="폰트 이미지 업로드"),
    outputs=[
        gr.Textbox(label="예측 결과", lines=15),
        gr.Image(type="pil", label="예시 text patch")
    ],
    title="폰트 분류기 (Patch Voting)",
    description="이미지를 드래그해서 업로드하면 patch voting으로 폰트를 예측합니다."
)


# ----------------------------
# 실행
# ----------------------------
if __name__ == "__main__":
    demo.launch()