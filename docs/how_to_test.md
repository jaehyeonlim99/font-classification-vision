## 폰트 분류 및 필기자 식별 모델 테스트 가이드라인

### 1. 개요

본 문서는 `머신 비전 프로젝트`에서 손글씨 폰트 분류 및 필기자 식별 모델의 성능을 객관적으로 평가하기 위한 표준 테스트 절차를 정의합니다.  
모든 팀원은 본 가이드라인에 명시된 디렉터리 구조, 모델/전처리 인터페이스, 평가 함수 동작 방식을 준수하여 모델을 구현하고 평가해야 합니다.

### 2. 프로젝트 디렉터리 구조

루트(`머신 비전 프로젝트`) 기준으로 테스트와 관련된 최소 디렉터리 구조는 아래와 같습니다.

```ASCII
머신 비전 프로젝트/
├── model/
│   ├── __init__.py
│   ├── dummy.py          # 예시용 더미 모델 (PyTorch)
│   └── preprocess.py     # 공통 전처리 인터페이스(get_*_transform) 정의
├── test/
│   ├── __init__.py
│   ├── test.py           # 표준 평가 함수(evaluate_model)만 포함 — 수정하지 않음
│   ├── run_test.py       # 평가 실행 진입점 (모델/데이터 경로 등 여기서만 수정 가능)
│   └── plot_performance.py  # confusion matrix / 클래스별 성능 시각화
└── generate_data/
    └── data/
        └── test/         # 8종 폰트별 폴더로 구분된 테스트 이미지 (ImageFolder 구조)
```

- `test/test.py`: 평가 로직(`evaluate_model`)만 담고 있으며, **임의로 수정하지 않는** 파일입니다.  
    어디에서 실행하든 `model` 패키지를 import 하기 위해 상위 디렉터리를 `sys.path`에 추가합니다.
- `test/run_test.py`: 실제로 평가를 실행할 때의 진입점입니다. 사용할 모델, 데이터셋 경로, 시각화 저장 경로 등을 이 파일에서만 바꿀 수 있습니다.
- 위와 같은 디렉터리 구조를 유지해야 import 및 실행에 문제가 발생하지 않습니다.

### 3. 전처리 인터페이스 규격 (`model/preprocess.py`)

전처리 코드는 모델과 테스트 코드 양쪽에서 공통으로 사용할 수 있도록 `model/preprocess.py`에 정의합니다.  
이 파일은 다음과 같은 함수를 제공해야 합니다.

- `get_base_transform()`
    - 학습/테스트 공통으로 사용하는 기본 전처리 파이프라인을 반환합니다.
    - 기본 구현 예시:

        ```python
        from torchvision import transforms

        def get_base_transform():
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225]),
            ])
        ```

- `get_train_transform()`
    - 학습 시 사용되는 전처리를 반환합니다.
    - 현재는 `get_base_transform()`을 그대로 사용하지만, 추후 데이터 증강(RandomCrop, RandomHorizontalFlip 등)을 추가할 경우 이 함수에만 변경을 가하면 됩니다.

- `get_test_transform()`
    - 테스트 시 사용되는 전처리를 반환합니다.
    - `test/test.py`의 `evaluate_model`은 항상 이 함수만 호출하도록 되어 있으므로,  
        전처리 로직을 바꾸고 싶을 때는 `model/preprocess.py`만 수정하면 되고 테스트 코드는 수정할 필요가 없습니다.

요약하면, 테스트 코드는 “`from model import preprocess` + `preprocess.get_test_transform()`만 알고 있고, 전처리 구현 세부 내용은 `preprocess.py`에 캡슐화되어 있다는 점이 핵심입니다.

### 4. 모델 인터페이스 규격 (`model/dummy.py` 기준)

모델 제작자는 프레임워크(PyTorch, TensorFlow 등)에 상관없이 다음 인터페이스를 만족하는 형태로 구현하는 것을 권장합니다.

- 입력 규격
    - 전처리를 통과한 이미지 텐서: \( (B, 3, 224, 224) \)
    - \(B\): 배치 크기

- 출력 규격
    - 로짓(logits): \( (B, N) \) 텐서 (\(N\)은 클래스 개수, 현재 8)
    - 필요 시 `softmax`를 적용해 확률 분포로 변환

- 필수 요소(권장)
    - PyTorch 기준 예시:
        - `class YourModel(nn.Module):`
            - `def forward(self, x):`  
                → `evaluate_model`에서 직접 호출하는 메서드 (logits 반환)
            - 선택: `def predict(self, x):`  
                → `softmax` 적용 후 `numpy` 배열로 반환하는 Keras 스타일 인터페이스 (추가 응용 시 편의를 위해)

현재 제공된 예시 모델 `DummyHandwritingModel`는 `forward(x)`에서 logits 를 반환하고,  
별도의 `predict(x)` 메서드는 `softmax` + `.cpu().numpy()`까지 수행하는 편의 함수로 구현되어 있습니다.

### 5. 평가 함수 구성 (`test/test.py`의 `evaluate_model`)

`test/test.py`에 구현된 `evaluate_model` 함수는 모든 모델에 공통으로 적용 가능한 표준 평가 루틴입니다.  
주요 흐름은 다음과 같습니다.

- 입력
    - `model`: `torch.nn.Module` 기반 모델 인스턴스 (`forward`에서 logits 반환)
    - `dataset_path`: 테스트 데이터셋 경로 (ImageFolder 구조, 예: `../generate_data/data/test`)
    - `criterion`: 손실 함수 (기본값: `nn.CrossEntropyLoss()`)

- 1) 전처리 및 데이터 로드
    - `transform = preprocess.get_test_transform()`  
        → `model/preprocess.py`에서 정의한 테스트용 전처리를 사용
    - `test_set = datasets.ImageFolder(root=dataset_path, transform=transform)`
    - `test_loader = DataLoader(test_set, batch_size=32, shuffle=False)`

- 2) 디바이스 및 평가 모드 설정
    - `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
    - `model.to(device)` 후 `model.eval()` 호출

- 3) 추론 및 지표 계산
    - `torch.no_grad()` 컨텍스트 안에서 배치 단위로 다음을 수행:
        - `logits = model(images)`
        - `probs = softmax(logits, dim=1)` → `numpy` 배열로 변환
        - `loss = criterion(logits, labels)` → 전체 loss 에 누적
        - 배치별 추론 시간 측정 후, 이미지 1장당 평균 추론 시간 리스트에 저장
        - `preds = argmax(probs, axis=1)` → 전체 예측/정답 레이블 리스트에 누적
        - 예측이 틀린 샘플에 대해
            - `true_label`, `pred_label`, `confidence(max prob)`를 담은 dict 를 `misclassified_log`에 기록

- 4) 클래스별 TP/TN/FP/FN 및 혼동 행렬
    - `confusion_matrix(all_labels, all_preds)`로 전체 혼동 행렬 계산
    - 각 클래스에 대해 다음 값 계산:
        - `TP`, `TN`, `FP`, `FN`
    - `class_detailed_metrics` 딕셔너리에 저장

- 5) 반환값 (`performance_dict`)
    - `average_loss`: 전체 테스트 셋 평균 loss
    - `total_accuracy`: 전체 정확도
    - `f1_score_weighted`: 클래스 불균형을 고려한 가중치 F1-score
    - `class_detailed_metrics`: 클래스별 TP/TN/FP/FN 정보
    - `confusion_matrix`: 혼동 행렬 (JSON 직렬화를 위해 리스트 형태)
    - `avg_inference_time_sec`: 이미지 1장당 평균 추론 시간(초)
    - `misclassified_log`: 오분류 샘플 상세 로그
    - `gpu_info`: 사용한 GPU 이름 또는 `"CPU"`

이 `performance_dict`는 콘솔 출력, 로그 저장, `plot_performance.py`를 이용한 시각화 등 다양한 용도로 재사용됩니다.

### 6. 시각화 모듈 (`test/plot_performance.py`)

`plot_performance.py`는 `evaluate_model`이 반환한 `results` 딕셔너리를 입력으로 받아 성능 지표를 시각화합니다.

- 입력
    - `results`: `performance_dict` 형식
    - `save_path_prefix`: 결과 이미지를 저장할 디렉터리 경로 (예: `"../res"`)
    - `show_plots`: `True`면 화면에 플롯을 표시, `False`면 파일만 저장

- 주요 출력
    - `confusion_matrix_heatmap.png`  
        → 혼동 행렬을 히트맵 형태로 표시
    - `class_metrics_bar_chart.png`  
        → 클래스별 TP/FP/FN 카운트를 막대 그래프로 시각화

### 7. 실행 방법 (Execution)

테스트 실행은 프로젝트 루트(= `머신 비전 프로젝트`)에서 `run_test.py` 를 진입점으로 사용합니다.

1. 터미널에서 프로젝트 루트로 이동:

    ```bash
    cd "머신 비전 프로젝트"
    ```

2. 다음 명령으로 평가 실행:

    ```bash
    python -m test.run_test
    ```

3. 실행 결과:
    - 콘솔에 다음 항목들이 출력됩니다.
        - `average_loss`
        - `total_accuracy`
        - `f1_score_weighted`
        - `class_detailed_metrics`
        - `avg_inference_time_sec`
        - `confusion_matrix`
        - `gpu_info`
    - 오분류된 샘플의 총 개수(`len(misclassified_log)`) 및 GPU 정보가 함께 출력됩니다.
    - `run_test.py`에서 호출하는 `plot_performance.plot_performance(...)`에 의해, 프로젝트 루트 기준 `res/` 디렉터리에 시각화 이미지가 생성됩니다.

다른 모델을 평가하고 싶을 때: `test/test.py`는 수정하지 않고, `test/run_test.py` 상단의 모델 import와 `if __name__ == "__main__":` 블록 안의 `model = ...` 한 줄만 교체하면 됩니다.  
GPU가 없는 환경에서는 CPU로 자동 전환되며, 가능하다면 CUDA 환경에서 실행하는 것을 권장합니다.

### 8. 대상 클래스 정보 (8개 폰트)

평가 시 사용되는 라벨 인덱스는 `ImageFolder`가 폴더명을 사전순으로 정렬한 순서를 따르며, 현재는 아래 8개 폰트가 대상입니다.

1. 나눔손글씨 아줌마 자유 (`NanumAJumMaJaYu`)
2. 나눔손글씨 대한민국 열사체 (`NanumDaeHanMinGugYeorSaCe`)
3. 나눔손글씨 강인한 위로 (`NanumGangInHanWiRo`)
4. 나눔손글씨 끄트머리체 (`NanumGgeuTeuMeoRiCe`)
5. 나눔손글씨 고려글꼴 (`NanumGoRyeoGeurGgor`)
6. 나눔손글씨 상해찬미체 (`NanumSangHaeCanMiCe`)
7. 나눔손글씨 소미체 (`NanumSoMiCe`)
8. 나눔손글씨 와일드 (`NanumWaIrDeu`)

새로운 모델을 추가할 때는 이 클래스 순서에 맞춰 출력 차원(`num_classes`)과 라벨 매핑을 유지해야, 테스트 코드 및 시각화 결과와 일관성이 보장됩니다.