from pathlib import Path
from PIL import Image
import numpy as np


# ----------------------------
# 경로 설정
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"

INPUT_DIR = DATASET_DIR / "synthetic_dataset"
OUTPUT_DIR = DATASET_DIR / "synthetic_patch_dataset"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# 설정
# ----------------------------
PATCH_SIZE = 224
STRIDE = 112

MIN_DARK_RATIO = 0.02   # 글자로 판단되는 픽셀 비율 최소값
OFFSET = 25             # 배경 추정 밝기보다 이만큼 더 어두워야 글자로 간주

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ----------------------------
# patch 시작 좌표 계산
# ----------------------------
def get_positions(length, patch_size, stride):
    if length <= patch_size:
        return [0]

    positions = list(range(0, length - patch_size + 1, stride))
    last_pos = length - patch_size

    if positions[-1] != last_pos:
        positions.append(last_pos)

    return positions


# ----------------------------
# patch가 글자를 충분히 포함하는지 검사
# ----------------------------
def is_text_patch(patch, min_dark_ratio=0.02, offset=25):
    gray = np.array(patch.convert("L"))

    # patch 내부의 밝은 픽셀들을 기준으로 배경 밝기 추정
    bg_est = np.percentile(gray, 90)

    # 배경보다 offset 이상 어두운 픽셀만 글자로 간주
    threshold = bg_est - offset
    dark_mask = gray < threshold
    dark_ratio = np.mean(dark_mask)

    return dark_ratio >= min_dark_ratio


# ----------------------------
# 이미지 한 장에서 patch 추출
# ----------------------------
def extract_patches(image, patch_size=224, stride=112):
    width, height = image.size

    x_positions = get_positions(width, patch_size, stride)
    y_positions = get_positions(height, patch_size, stride)

    patches = []
    coords = []

    for y in y_positions:
        for x in x_positions:
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            coords.append((x, y))

    return patches, coords


# ----------------------------
# 전체 synthetic patch 생성
# ----------------------------
def main():
    if not INPUT_DIR.exists():
        print(f"[오류] 입력 폴더가 없음: {INPUT_DIR}")
        return

    class_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])

    if not class_dirs:
        print(f"[오류] 클래스 폴더가 없음: {INPUT_DIR}")
        return

    total_saved = 0
    total_skipped = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        save_dir = OUTPUT_DIR / class_name
        save_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        )

        if not image_files:
            print(f"[경고] 이미지 없음: {class_dir}")
            continue

        print(f"{class_name} 시작")

        class_saved = 0
        class_skipped = 0

        for img_path in image_files:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[스킵] 이미지 열기 실패: {img_path.name} | {e}")
                continue

            patches, coords = extract_patches(
                image,
                patch_size=PATCH_SIZE,
                stride=STRIDE
            )

            stem = img_path.stem

            for idx, (patch, (x, y)) in enumerate(zip(patches, coords)):
                if not is_text_patch(
                    patch,
                    min_dark_ratio=MIN_DARK_RATIO,
                    offset=OFFSET
                ):
                    class_skipped += 1
                    total_skipped += 1
                    continue

                save_name = f"{stem}_patch_{idx:02d}_x{x}_y{y}.png"
                patch.save(save_dir / save_name)
                class_saved += 1
                total_saved += 1

        print(f"{class_name} 완료 - 저장된 patch 수: {class_saved}, 제외된 patch 수: {class_skipped}")

    print(f"synthetic_patch_dataset 생성 완료 - 전체 저장 patch 수: {total_saved}, 전체 제외 patch 수: {total_skipped}")


if __name__ == "__main__":
    main()