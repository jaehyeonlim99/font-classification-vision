from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import numpy as np



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"

INPUT_DIR = DATASET_DIR / "real_screen_photo_raw"
OUTPUT_DIR = DATASET_DIR / "real_patch_dataset"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PATCH_SIZE = 512
STRIDE = 256

MIN_DARK_RATIO = 0.02
OFFSET = 25

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def get_positions(length, patch_size, stride):
    if length <= patch_size:
        return []

    positions = list(range(0, length - patch_size + 1, stride))

    last_pos = length - patch_size
    if positions[-1] != last_pos:
        positions.append(last_pos)

    return positions


def is_text_patch(
    patch,
    min_mean_brightness=80,
    min_edge_ratio=0.008,
    min_component_area=100,
    min_component_count=3
):
    img = np.array(patch)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1) 너무 어두운 패치는 제거
    if np.mean(gray) < min_mean_brightness:
        return False

    # 2) patch 내부 밝기 기준으로 어두운 부분 추출
    bg_est = np.percentile(gray, 90)
    threshold = max(0, int(bg_est - 25))

    binary = (gray < threshold).astype(np.uint8) * 255

    # 3) 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 4) edge 비율 확인
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = np.mean(edges > 0)

    if edge_ratio < min_edge_ratio:
        return False

    # 5) 연결 성분 분석
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    large_components = 0
    for i in range(1, num_labels):  # 0은 배경
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            large_components += 1

    if large_components < min_component_count:
        return False

    return True


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


def main():

    class_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])

    total_saved = 0

    for class_dir in class_dirs:

        class_name = class_dir.name
        save_dir = OUTPUT_DIR / class_name
        save_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        )

        print(f"{class_name} 시작")

        class_saved = 0

        for img_path in image_files:

            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue

            width, height = image.size

            # 너무 작은 이미지는 skip
            if width < PATCH_SIZE or height < PATCH_SIZE:
                continue

            patches, coords = extract_patches(image)

            stem = img_path.stem

            for idx, (patch, (x, y)) in enumerate(zip(patches, coords)):

                if not is_text_patch(patch):
                    continue

                save_name = f"{stem}_patch_{idx}_x{x}_y{y}.png"

                patch.save(save_dir / save_name)

                class_saved += 1
                total_saved += 1

        print(f"{class_name} 완료 - 저장된 patch 수: {class_saved}")

    print(f"real_patch_dataset 생성 완료 - 전체 patch 수: {total_saved}")


if __name__ == "__main__":
    main()