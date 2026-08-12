from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# 경로 설정
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models/checkpoints/resnet18_font_cls_best.pth"
DATASET_DIR = PROJECT_ROOT / "dataset/real_screen_photo_raw"


# ----------------------------
# patch 설정
# ----------------------------
PATCH_SIZE = 512
STRIDE = 256

IMAGE_SIZE = 224


# ----------------------------
# transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ----------------------------
# patch 생성
# ----------------------------
def extract_patches(image):

    w, h = image.size
    patches = []

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):

            patch = image.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            patches.append(patch)

    return patches


# ----------------------------
# 모델 로드
# ----------------------------
def load_model():

    checkpoint = torch.load(MODEL_PATH, map_location="cuda")

    class_names = checkpoint["class_names"]

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()
    model.eval()

    return model, class_names


# ----------------------------
# patch voting
# ----------------------------
def predict_image(model, image):

    patches = extract_patches(image)

    votes = []

    with torch.no_grad():

        for patch in patches:

            x = transform(patch).unsqueeze(0).cuda()

            out = model(x)

            prob = torch.softmax(out, dim=1)

            conf, pred = torch.max(prob, 1)

            # confidence filtering
            if conf.item() < 0.4:
                continue

            votes.append(pred.item())

    if len(votes) == 0:
        return None

    vote = Counter(votes).most_common(1)[0][0]

    return vote


# ----------------------------
# 전체 평가
# ----------------------------
def evaluate():

    model, class_names = load_model()

    y_true = []
    y_pred = []

    for class_idx, class_name in enumerate(class_names):

        folder = DATASET_DIR / class_name

        images = list(folder.glob("*"))

        for img_path in images:

            img = Image.open(img_path).convert("RGB")

            pred = predict_image(model, img)

            if pred is None:
                continue

            y_true.append(class_idx)
            y_pred.append(pred)

            print(f"{img_path.name} → {class_names[pred]}")


    # accuracy
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print("\nAccuracy:", acc)

    # classification report
    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Font Classification Confusion Matrix")

    plt.show()


if __name__ == "__main__":
    evaluate()