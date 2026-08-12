from pathlib import Path
import shutil
import random


# ----------------------------
# 경로 설정
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"

SYNTHETIC_DIR = DATASET_DIR / "synthetic_patch_dataset"
REAL_DIR = DATASET_DIR / "real_patch_dataset"
OUTPUT_DIR = DATASET_DIR / "merged_patch_dataset"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# 설정
# ----------------------------
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
RANDOM_SEED = 42

# real 1개당 synthetic 몇 개를 뽑을지
SYNTHETIC_PER_REAL = 3


def get_image_files(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def clear_output_dir(folder: Path):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def copy_files(files, dst_dir: Path, prefix: str):
    copied = 0
    for i, src_path in enumerate(files):
        new_name = f"{prefix}_{i:05d}{src_path.suffix.lower()}"
        shutil.copy2(src_path, dst_dir / new_name)
        copied += 1
    return copied


def main():
    random.seed(RANDOM_SEED)

    if not SYNTHETIC_DIR.exists():
        print(f"[오류] synthetic 폴더 없음: {SYNTHETIC_DIR}")
        return

    if not REAL_DIR.exists():
        print(f"[오류] real 폴더 없음: {REAL_DIR}")
        return

    clear_output_dir(OUTPUT_DIR)

    class_names = sorted([
        d.name for d in REAL_DIR.iterdir()
        if d.is_dir() and (SYNTHETIC_DIR / d.name).exists()
    ])

    if not class_names:
        print("[오류] 공통 클래스 폴더가 없음")
        return

    total_real = 0
    total_syn = 0
    total_merged = 0

    print(f"[설정] synthetic : real = {SYNTHETIC_PER_REAL} : 1")
    print()

    for class_name in class_names:
        real_class_dir = REAL_DIR / class_name
        syn_class_dir = SYNTHETIC_DIR / class_name
        out_class_dir = OUTPUT_DIR / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        real_files = get_image_files(real_class_dir)
        syn_files = get_image_files(syn_class_dir)

        real_count = len(real_files)
        syn_count = len(syn_files)

        if real_count == 0:
            print(f"[경고] {class_name}: real patch 없음, 스킵")
            continue

        target_syn_count = real_count * SYNTHETIC_PER_REAL
        sample_syn_count = min(target_syn_count, syn_count)

        sampled_syn_files = random.sample(syn_files, sample_syn_count)

        copied_real = copy_files(real_files, out_class_dir, "real")
        copied_syn = copy_files(sampled_syn_files, out_class_dir, "syn")

        class_total = copied_real + copied_syn

        total_real += copied_real
        total_syn += copied_syn
        total_merged += class_total

        print(
            f"{class_name} 완료 | "
            f"real: {copied_real} | "
            f"synthetic: {copied_syn} | "
            f"merged: {class_total}"
        )

    print()
    print("merged_patch_dataset 생성 완료")
    print(f"전체 real 수: {total_real}")
    print(f"전체 synthetic 수: {total_syn}")
    print(f"전체 merged 수: {total_merged}")


if __name__ == "__main__":
    main()