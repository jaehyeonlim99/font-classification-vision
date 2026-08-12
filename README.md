# Font Classification Vision

문서 이미지에서 폰트를 분류하기 위한 ResNet18 기반 computer vision 프로젝트입니다. synthetic font image와 real document patch를 활용하고, 전체 이미지에서 text patch를 추출한 뒤 patch voting으로 최종 폰트를 예측하는 구조로 설계했습니다.

## Problem

문서 전체 이미지를 한 번에 분류하면 배경, 여백, 이미지 품질, 글자 밀도 차이 때문에 폰트 특징이 희석될 수 있습니다. 이 프로젝트는 문서 이미지를 patch 단위로 나누고, 텍스트가 포함된 patch만 선별해 폰트별 특징을 더 안정적으로 학습/추론하는 것을 목표로 했습니다.

## Solution

- 8종 font classification task 정의
- synthetic dataset 생성으로 학습 데이터 확보
- real document image에서 text patch 추출
- synthetic/real patch dataset merge
- ImageNet pretrained ResNet18 fine-tuning
- patch-level prediction 후 majority voting으로 image-level font 결정
- Gradio 기반 demo app 구성

## Tech Stack

- Python
- PyTorch
- Torchvision ResNet18
- OpenCV
- PIL
- Gradio

## Structure

```text
.
├─ app
│  └─ app.py
├─ src
│  ├─ data
│  │  ├─ make_synthetic_dataset.py
│  │  ├─ make_synthetic_patch_dataset.py
│  │  ├─ make_real_patch_dataset.py
│  │  └─ make_merged_patch_dataset.py
│  ├─ train
│  │  └─ train_resnet18.py
│  └─ eval
│     └─ evaluate_patch_voting.py
├─ results
│  ├─ class_metrics_bar_chart.png
│  └─ confusion_matrix_heatmap.png
├─ docs
└─ requirements.txt
```

## Core Ideas

| Step | Description |
| --- | --- |
| Data generation | TTF font를 이용해 synthetic text image 생성 |
| Patch extraction | 큰 문서 이미지를 일정 크기의 patch로 분할 |
| Text patch filtering | 밝기, edge ratio, connected component 조건으로 텍스트 patch 선별 |
| Training | ResNet18 classifier fine-tuning |
| Inference | patch별 예측 결과를 voting해 최종 font 결정 |

## Quick Start

```bash
pip install -r requirements.txt
python app/app.py
```

학습을 다시 수행하려면 dataset을 준비한 뒤 아래 스크립트를 실행합니다.

```bash
python src/train/train_resnet18.py
```

## Repository Scope

Included:

- training code
- inference/demo code
- dataset generation scripts
- patch voting evaluation code
- result charts
- project docs

Excluded:

- `.venv`
- generated dataset
- TTF font files
- model checkpoint by default

Model checkpoint 공개가 필요하면 font license와 dataset 공개 범위를 확인한 뒤 Git LFS로 관리하는 것이 좋습니다.
