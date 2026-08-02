# 의류 수요 정보 예측을 위한 멀티모달 딥 뉴럴 네트워크

**저자:** 김동주, 이민식  
**소속:** 한양대학교 인공지능융합학과, 전자공학과

[2022 대한전자공학회 추계학술대회](https://conf.theieie.org/2022f/)에서 발표

## Overview

본 프로젝트는 온라인 쇼핑 데이터에서 시각적(이미지)과 텍스트 정보를 결합하여 의류 상품의 수요 정보(조회수, 누적판매량)를 예측하는 멀티모달 딥 뉴럴 네트워크를 제안합니다. 모델 아키텍처는 다음을 통합합니다:

- **ResNet18** (사전학습 모델): 이미지 특징 추출
- **다국어 BERT**: 상품명 인코딩
- **테이블형 메타데이터** (성별, 가격, 카테고리): 맥락 정보
- **멀티태스크 학습** (4개 예측 헤드):
  1. 선호 성별 (3 클래스)
  2. 선호 연령대 (7 클래스)
  3. 조회수
  4. 누적판매량

데이터셋은 무신사 온라인 쇼핑몰에서 웹 스크래핑하여 구성되며, 제품 이미지, 상품명, 가격 및 메타데이터와 스크래핑한 라벨 정보를 포함합니다. 자세한 아키텍처, 손실함수, 구현 상세는 [docs/SDD.md](docs/SDD.md)를 참고하세요.

## Repository Structure

```
kim2022multi/
├── README.md                    (본 파일)
├── LICENSE
├── .gitignore
├── requirements.txt
├── Materials/
│   ├── paper.pdf                  (학회 논문)
│   ├── poster.png / poster.pdf    (학회 포스터)
│   ├── Additional experiment1.png    (훈련 곡선: 정확도)
│   └── Additional experiment2.png    (훈련 곡선: 회귀 손실)
├── docs/
│   ├── SDD.md                      (소프트웨어 설계 문서 — 아키텍처, 모듈, 데이터 흐름)
│   └── TC.md                       (테스트 케이스 — 70+ 검증 항목)
└── src/
    ├── config.py                   (하이퍼파라미터: batch_size=64, num_epochs=3000, lr=1e-4 등)
    ├── models/
    │   ├── __init__.py
    │   └── resnet_pre_trained.py    (ResNet18 + BERT 융합 멀티태스크 모델)
    ├── data/
    │   └── shopping_dataset.py      (데이터셋 로더: 이미지 파일명 파싱 및 BERT 특징 통합)
    ├── utils/
    │   ├── bert_features.py         (BERT 토큰화 및 특징 추출)
    │   ├── training_utils.py        (포칼 손실, 멀티태스크 손실, 정확도 메트릭)
    │   └── io_utils.py              (체크포인트/결과 저장, 디렉토리 생성)
    ├── train.py                     (메인 훈련 루프: 4개 헤드 결합 학습)
    └── train_single_task.py         (단일 태스크 어블레이션: 헤드별 평가)
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (GPU 훈련용)

### Setup

```bash
# 저장소 클론 및 디렉토리 이동
git clone <저장소_url>
cd kim2022multi

# 가상 환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## Environment Configuration

Configuration is managed through `src/config.py`, which supports environment variables loaded from a `.env` file. This allows you to override default hyperparameters without modifying source code.

### Setup .env File

1. Copy the example configuration:

```bash
cp .env.example .env
```

2. Customize values in `.env` as needed:

```env
# Training Configuration
BATCH_SIZE=64
NUM_EPOCHS=3000
LR=0.0001
WEIGHT_DECAY=1e-5
CLIP_NORM=5

# Focal Loss (for age prediction)
FOCAL_ALPHA=1
FOCAL_GAMMA=2

# Multi-Task Loss Weights
LOSS_ALPHA=0.01  # Weight for view count regression
LOSS_BETA=0.01   # Weight for sales volume regression

# CUDA Device Selection
CUDA_VISIBLE_DEVICES=0

# Optional: Data and Output Paths (uses defaults if not set)
# DATA_PATH=/path/to/dataset/category_all_ver2_20221002_words_125_aug
# CSV_PATH=/path/to/dataset/goodsNum_clothing_name_20221002.csv
# CHECKPOINT_DIR=/path/to/checkpoints
# RESULTS_DIR=/path/to/results
```

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE` | 64 | Batch size per GPU |
| `NUM_EPOCHS` | 3000 | Number of training epochs |
| `LR` | 0.0001 | Adam optimizer learning rate |
| `WEIGHT_DECAY` | 1e-5 | L2 regularization coefficient |
| `CLIP_NORM` | 5 | Gradient clipping threshold |
| `FOCAL_ALPHA` | 1 | Focal loss α (age prediction) |
| `FOCAL_GAMMA` | 2 | Focal loss γ (age prediction) |
| `LOSS_ALPHA` | 0.01 | Multi-task weight for view prediction |
| `LOSS_BETA` | 0.01 | Multi-task weight for sales prediction |
| `CUDA_VISIBLE_DEVICES` | 0 | GPU device ID(s) |
| `DATA_PATH` | `dataset/category_all_ver2_20221002_words_125_aug/` | Dataset image directory |
| `CSV_PATH` | `dataset/goodsNum_clothing_name_20221002.csv` | Product metadata CSV |
| `CHECKPOINT_DIR` | `checkpoints/` | Directory for model checkpoints |
| `RESULTS_DIR` | `results/` | Directory for training results |

All values are loaded automatically when you run `src/config.py`. If `.env` is not present, hardcoded defaults are used.

## Dataset

데이터셋은 온라인 패션 쇼핑몰 데이터로 구성되며, 저장소에 포함되지 않습니다.

**기대되는 구조**:
```
dataset/
├── category_all_ver2_20221002_words_125_aug/
│   ├── 0/*.png   (125×125 RGB images)
│   ├── 1/*.png
│   └── ...
└── goodsNum_clothing_name_20221002.csv  (상품 메타데이터)
```

**Filename Format:** 각 이미지 파일명은 다음 형식을 따릅니다:
```
{idx}_{goodsNum}_{sex}_{best_sex}_{best_age}_{view}_{category}_{price}_{sales}_{extra1}_{extra2}.png
```

자세한 내용은 [docs/SDD.md - 데이터 흐름 및 라벨 인코딩](docs/SDD.md#4-data-flow--label-encoding)을 참고하세요.

## Usage

### Multi-Task Training

4개 헤드를 결합 손실로 동시 훈련:

```bash
python -m src.train
```

**출력:**
- 모델 체크포인트: `checkpoints/...model_state_dict.pt`
- 훈련 로그: `results/AccLoss2.txt`

**설정:** `.env` 파일을 통해 하이퍼파라미터 조정 (예: `BATCH_SIZE`, `NUM_EPOCHS`, `LR`). 또는 `src/config.py`의 기본값을 직접 수정.

### Single-Task Evaluation (Ablation Study)

각 예측 헤드를 독립적으로 평가:

```bash
# 선호 성별 분류기만 훈련
python -m src.train_single_task --analysis best_sex

# 선호 연령대 분류기만 훈련 (클래스 불균형 처리용 포칼 손실)
python -m src.train_single_task --analysis best_age

# 조회수 회귀만 훈련
python -m src.train_single_task --analysis view

# 누적판매량 회귀만 훈련
python -m src.train_single_task --analysis sales
```

**주의:** 스크립트명("train_single_task")과 달리 이 모드는 여전히 BERT feature을 사용합니다. 이는 단일 태스크 어블레이션(한 번에 하나의 헤드)이지 텍스트 제거 설정은 아닙니다. 자세한 내용은 [docs/SDD.md](docs/SDD.md#8-알려진-한계-및-편차)를 참고하세요.

## Experimental Results

**테스트 셋 성능** (포스터, 3,000 에포크):

| 지표 | 제안 방법 | 텍스트 없음 어블레이션 |
|------|----------|----------------------|
| 선호 성별 정확도 | 84.5% | 83.9% |
| 선호 연령대 정확도 | 73.7% | 73.5% |
| 조회수 MSE | 0.092 | 0.090 |
| 누적판매량 MSE | 0.054 | 0.058 |

**제안된 멀티모달 방법**은 이미지와 텍스트 특징을 효과적으로 활용하여 4개 지표 중 3개(성별, 연령대, 판매량)에서 우수한 성능을 달성합니다.

전체 훈련 곡선과 모든 모델 변형에 대한 자세한 내용은 [Materials/Additional experiment1.png](Materials/Additional%20experiment1.png)과 [Materials/Additional experiment2.png](Materials/Additional%20experiment2.png)를 참고하세요.

## Configuration

`src/config.py`의 주요 하이퍼파라미터:

- `batch_size`: 64
- `num_epochs`: 3000
- `lr`: 0.0001 (Adam 옵티마이저)
- `weight_decay`: 1e-5
- **포칼 손실:** α=1, γ=2 (선호 연령대 분류의 클래스 불균형 처리)
- **멀티태스크 손실 가중치:**
  - 성별: 1.0 (교차 엔트로피)
  - 연령대: 1.0 (포칼 손실)
  - 조회수: 0.01 (MSE)
  - 판매량: 0.01 (MSE)

전체 아키텍처 상세는 [docs/SDD.md](docs/SDD.md)를 참고하세요.

## Technical Documentation

- **[docs/SDD.md](docs/SDD.md)** — 소프트웨어 설계 문서
  - 시스템 아키텍처 및 데이터 파이프라인
  - 상세 모듈 설명 (8개 파일)
  - 파일명 인코딩 스킴 (11필드 포맷)
  - 손실함수 유도
  - 논문 ↔ 코드 대응 맵핑
  - 알려진 한계 및 설계 결정사항

- **[docs/TC.md](docs/TC.md)** — 테스트 케이스
  - 유닛, 통합, 시스템, 회귀 테스트를 포함한 70+ 검증 항목
  - 버그 수정 검증 (가격 인덱스 일치성, BERT 특징 순서 등)
  - 체크포인트 저장/로드 검증
  - 멀티태스크 손실 공식 검증

## Troubleshooting

| 문제 | 해결책 |
|------|--------|
| `FileNotFoundError: model_weights/` 또는 `results/` | 디렉토리는 첫 실행 시 자동 생성됩니다. 쓰기 권한을 확인하세요. |
| `CUDA out of memory` | `src/config.py`에서 `batch_size` 감소 (기본값: 64). |
| `BertModel download fails` (오프라인) | 사전 다운로드: `python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-multilingual-cased')"` |
| ChromeDriver 버전 불일치 | ChromeDriver 버전을 Chrome/Chromium 브라우저 버전과 일치시키세요. |
| CSV 인코딩 에러 | `info.csv`가 **cp949** (한글, EUC-KR)로 인코딩되었는지 확인하세요. (무신사 내보내기 형식) |

## Poster

<img src="Materials/poster.png" width="100%"/>

## References

```bibtex
@inproceedings{kim2022multimodal,
  title={의류 수요 정보 예측을 위한 멀티모달 기반 딥 뉴럴 네트워크},
  author={Kim, Dongjoo and Lee, Minsik},
  booktitle={대한전자공학회 학술대회},
  year={2022},
  pages={788--791},
  organization={KIEE}
}
```

```
김동주, and 이민식. "의류 수요 정보 예측을 위한 멀티모달 기반 딥 뉴럴 네트워크." 대한전자공학회 학술대회 (2022): 788-791.
```

## License

자세한 내용은 [LICENSE](LICENSE)를 참고하세요.

## Contact

질문이나 문제 사항:
- **이메일:** dongjookim1541@gmail.com
- **소속:** 한양대학교 인공지능융합학과
