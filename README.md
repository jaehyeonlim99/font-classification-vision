# Font Classification Vision

Computer vision project for classifying document images into 8 font classes. The project uses a ResNet18 classifier trained with synthetic and real image patches, then aggregates patch-level predictions with voting to improve image-level font classification.

## Problem

Classifying a full document image directly can be unstable because background, margins, noise, text size, and layout may hide the actual font characteristics. This project approaches the task by extracting text-focused patches from document images and classifying them at patch level before producing the final image-level prediction.

## Solution

- Defined an 8-class font classification task
- Built synthetic font images to compensate for limited real data
- Combined synthetic and real patch datasets
- Trained an ImageNet-pretrained ResNet18 model
- Applied patch-level prediction and majority voting for image-level classification
- Built a Gradio demo for checking classification results on real images

## My Contribution

- Selected and trained a ResNet18-based model for 8-class font classification
- Built the training data composition by combining synthetic and real images
- Used an approximate synthetic-to-real ratio of 4:1 to increase training coverage while keeping real data in the loop
- Implemented patch extraction and filtering logic for document images
- Applied patch voting to aggregate patch-level predictions into a final image-level result
- Compared training and validation results to verify model performance
- Confirmed final performance around 96.5% validation accuracy and 1.0 accuracy on the real64 evaluation set

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
| Data generation | Generate synthetic text images from font files |
| Patch extraction | Split document images into fixed-size patches |
| Text patch filtering | Select text-heavy patches using brightness, edge, and component conditions |
| Training | Fine-tune a ResNet18 classifier |
| Inference | Vote across patch predictions to determine the final font class |

## Quick Start

```bash
pip install -r requirements.txt
python app/app.py
```

To retrain the model, prepare the dataset first and run:

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

- virtual environments
- generated datasets
- TTF font files
- model checkpoints by default

If model checkpoints are published later, font license and dataset sharing policy should be reviewed first. Large checkpoint files should be managed with Git LFS.
