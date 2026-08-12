## 폰트 분류 / 필기자 식별 모델 구현 가이드 (`model/`)

### 1. 개요

이 문서는 `머신 비전 프로젝트/model/` 폴더 안에 새로운 모델 코드를 어떻게 작성해야 하는지를 설명합니다.  
목표는 다음과 같습니다.

- 테스트 로직(`test/test.py`의 `evaluate_model`)과의 인터페이스를 깨지지 않게 유지하면서
- 누구나 자신의 모델을 `model/`에 추가하고
- 동일한 테스트 파이프라인(`python -m test.run_test`)으로 공정하게 평가할 수 있도록 하는 것

### 2. 모델 파일 위치와 이름

- 모든 모델 파일은 `머신 비전 프로젝트/model/` 디렉터리 아래에 둡니다.
- 예시:

```ASCII
model/
├── __init__.py
├── dummy.py          # 예시용 기본 모델
├── preprocess.py     # 공통 전처리 인터페이스
└── my_awesome_model.py   # 새로 만든 모델 파일
```

파일 이름은 자유롭게 정해도 되지만, 클래스 이름과 파일 이름이 어느 정도 연관되게 작성하는 것을 권장합니다.
예: `my_awesome_model.py` 안에 `class MyAwesomeModel(nn.Module): ...`

### 3. 공통 전처리 사용 (`model/preprocess.py`)

모델은 직접 전처리를 구현하기보다는, 공통 전처리 모듈(`model/preprocess.py`)를 사용하는 것을 추천합니다.

- `model/preprocess.py`는 다음 함수를 제공합니다.
    - `get_base_transform()` : 학습/테스트 공통 기본 전처리
    - `get_train_transform()` : 학습용 전처리 (augmentation 추가 가능)
    - `get_test_transform()` : 테스트용 전처리 (`test/test.py`가 사용하는 인터페이스)

학습 스크립트를 작성할 때는 보통 다음과 같이 사용하면 됩니다.

```python
from model import preprocess

train_transform = preprocess.get_train_transform()
test_transform = preprocess.get_test_transform()
```

> 중요: 테스트 파이프라인(`test/test.py`)은 항상 `get_test_transform()`을 사용하므로,  
> 테스트 시 입력 분포를 바꾸고 싶다면 `model/preprocess.py`만 수정하면 되고 `test.py`는 건드릴 필요가 없습니다.

### 4. 모델 인터페이스 규격 (PyTorch 기준)

현재 테스트 코드는 PyTorch 모델을 기준으로 작성되어 있습니다.  
새로운 모델을 만들 때는 다음 규칙을 지켜 주세요.

#### 4.1 클래스 정의

- `torch.nn.Module`을 상속한 클래스로 작성합니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self, num_classes: int = 8):
        super(MyAwesomeModel, self).__init__()
        # TODO: 여기서 레이어 정의 (Conv, FC, etc.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 순전파 정의
        # x: (B, 3, 224, 224)
        # return: (B, num_classes) 형태의 logits
        ...
```

#### 4.2 입력 / 출력 규격

- 입력 텐서
    - shape: `(B, 3, 224, 224)`
    - 전처리(`preprocess.get_*_transform()`)를 통과한 결과를 그대로 받는다고 가정합니다.

- 출력 텐서
    - shape: `(B, num_classes)`
    - `num_classes = 8` (기본: 8개 폰트 클래스)
    - 각 row는 클래스별 logit 값 (softmax 이전 값)입니다.

> `test/test.py`의 `evaluate_model`은 `logits = model(images)` 후 `torch.softmax(logits, dim=1)`을 직접 호출하므로,  
> `forward`에서 softmax를 적용하지 않는 것을 권장합니다.

#### 4.3 선택: `predict` 메서드 (편의용)

필수는 아니지만, 사용 편의를 위해 Keras 스타일의 `predict` 메서드를 구현할 수 있습니다.  
예시는 `dummy.py`를 참고하세요.

```python
import numpy as np

class MyAwesomeModel(nn.Module):
    ...

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """softmax 확률을 numpy 배열로 반환하는 편의 메서드"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()
```

이 메서드는 연구/데모 코드에서 바로 확률 값을 보고 싶을 때 유용하며,  
테스트 파이프라인(`evaluate_model`)은 사용하지 않아도 됩니다.

### 5. `evaluate_model`과의 연결 방식

실제 평가 실행은 `test/run_test.py` 에서 이루어지며, 이 파일만 수정해도 됩니다. 기본 흐름은 다음과 같습니다.

```python
# test/run_test.py
from test.test import evaluate_model
from test import plot_performance
from model.dummy import DummyHandwritingModel

if __name__ == "__main__":
    model = DummyHandwritingModel(num_classes=8)
    dataset_path = os.path.join(_root, "generate_data", "data", "test")
    results = evaluate_model(model, dataset_path)
    plot_performance.plot_performance(results, save_path_prefix=res_dir, show_plots=True)
    ...
```

새 모델을 테스트하려면 `run_test.py`에서 import와 model 인스턴스만 교체하면 됩니다.

```python
from model.my_awesome_model import MyAwesomeModel

if __name__ == "__main__":
    model = MyAwesomeModel(num_classes=8)
    ...
```

> 테스트 인터페이스 요약
>
> - `evaluate_model(model, dataset_path, criterion=None)` (`test.test`에 정의됨)
>   - `model`: `forward(images)`가 `(B, num_classes)` logits 를 반환하는 PyTorch 모델
>   - `dataset_path`: ImageFolder 구조의 테스트 데이터 경로
>   - `criterion`: 손실 함수 (기본 `CrossEntropyLoss`)
> - 반환값: 다양한 지표가 담긴 `performance_dict` (자세한 내용은 `docs/how_to_test.md` 참고)

### 6. 클래스(폰트) 개수 변경 시 주의사항

현재 프로젝트는 8개 폰트를 대상으로 합니다.  
만약 클래스 수를 변경하고 싶다면, 다음 항목들을 함께 수정해야 합니다.

1. 데이터셋 구조
   - `generate_data/data/train`, `generate_data/data/test` 아래의 폴더 구성이 실제 클래스 수와 일치해야 합니다.
2. 모델의 `num_classes`
   - `MyAwesomeModel(num_classes=새로운_개수)`
   - 마지막 FC 레이어 출력 차원도 맞게 수정
3. 테스트 실행부
   - `test/run_test.py`에서 사용하는 모델의 `num_classes` 인자 (예: `MyAwesomeModel(num_classes=새로운_개수)`)
   - `plot_performance.py`는 `results['class_detailed_metrics']`를 기반으로 동작하므로 클래스 수 증가에는 자동 대응하지만,  
     폰트 이름/설명을 문서에 따로 적어둔 경우(`docs/how_to_test.md`의 클래스 목록)는 수동으로 갱신해야 합니다.

가능하면, 이 프로젝트에서는 8개 폰트 구성을 기준으로 모델만 교체하는 것을 권장합니다.

### 7. 모델 구현 시 팁

- 전처리와 모델을 분리하세요.
  - 전처리는 항상 `model/preprocess.py`에서 관리하고, 모델은 “전처리된 텐서를 어떻게 처리할지”만 책임지도록 합니다.
- GPU/CPU 호환성
  - 학습 코드에서는 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` 패턴을 사용하고,
  - 모델과 입력 텐서를 항상 같은 디바이스로 옮겨서 사용하세요.
- `eval()` / `train()` 모드 전환
  - 학습 시에는 `model.train()`, 평가/테스트 시에는 `model.eval()`을 호출해야 BatchNorm, Dropout 등이 올바르게 동작합니다.
  - `evaluate_model` 내부에서 이미 `model.eval()`을 호출하므로, 테스트 시에는 별도로 설정할 필요가 없습니다.

### 8. 요약 체크리스트

새 모델을 `model/`에 추가할 때, 아래 항목을 만족하면 테스트 파이프라인과 잘 연동됩니다.

- [ ] `model/` 디렉터리 아래에 새 파일을 만들었다. (예: `my_awesome_model.py`)
- [ ] `torch.nn.Module`을 상속한 모델 클래스를 정의했다.
- [ ] `forward(x)`는 입력 `(B, 3, 224, 224)` → 출력 `(B, num_classes)` logits를 반환한다.
- [ ] 테스트할 때 `test/run_test.py`에서 내 모델 클래스를 import 하고, `model = ...` 인스턴스만 교체했다.
- [ ] 전처리는 `model/preprocess.py`의 `get_test_transform()`을 사용한다. (테스트 코드 수정 없이 전처리 변경 가능)
- [ ] 클래스 수(폰트 개수)를 변경했다면, 데이터셋 구조와 모델 출력 차원, 관련 문서를 함께 갱신했다.

위 조건을 만족하면, 터미널에서 다음 명령만으로 기존 모델과 동일한 규격으로 성능 비교를 할 수 있습니다.

```bash
python -m test.run_test
```

모델을 바꿔가며 다양한 아키텍처를 실험하되, 인터페이스와 규격은 이 문서에 맞춰 유지해 주세요.
