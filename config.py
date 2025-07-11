import torch
import os 
# ─────────────────────────────────────────────────────────────────────────────
# 1. 재현성 & 로깅 설정
# ─────────────────────────────────────────────────────────────────────────────
SEED         = 42
LOGGER       = 'wandb'            # 'tensorboard' | 'wandb' | None
PROJECT_NAME = 'xformer'          # W&B project 이름
RUN_NAME     = 'exp1'             # W&B run 이름 또는 TensorBoard sub디렉터리

# ─────────────────────────────────────────────────────────────────────────────
# 2. 데이터 설정 (Data Settings)
# ─────────────────────────────────────────────────────────────────────────────
# 경로
TRAIN_DATA_PATH = "../test/cleaned_train_3mth.pkl"
VAL_DATA_PATH   = "../test/cleaned_valid_3mth.pkl"
TEST_DATA_PATH  = "../test/cleaned_test_3mth.pkl"

# 파라미터
MAX_LEN         = 120            # 최대 시퀀스 길이
PATCH_RADIUS_KM = 5.0            # 컨텍스트 맵 반경 (km)
TIME_STEP_MINS  = 10             # 타임스텝 간격 (분)
CONTEXT_CHANNEL = 3              # 맵의 채널 수
PATCH_SIZE      = 64             # 패치 해상도

# 위경도 정규화 범위
LAT_MIN, LAT_MAX = 55.5, 58.0
LON_MIN, LON_MAX = 10.3, 13.0

# ─────────────────────────────────────────────────────────────────────────────
# 3. 모델 하이퍼파라미터 (Model Hyperparameters)
# ─────────────────────────────────────────────────────────────────────────────
# 4-hot bin 크기
LAT_SIZE = 250
LON_SIZE = 270
SOG_SIZE = 30
COG_SIZE = 72

# 임베딩 차원
N_LAT_EMBD = 256
N_LON_EMBD = 256
N_SOG_EMBD = 128
N_COG_EMBD = 128

# 트랜스포머 설정
D_MODEL        = N_LAT_EMBD + N_LON_EMBD + N_SOG_EMBD + N_COG_EMBD
N_HEAD         = 8
NUM_LAYERS_ENC = 3
NUM_LAYERS_FUS = 3
NUM_LAYERS_DEC = 3 
DIM_FF         = 512
DROPOUT        = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# 4. 옵티마이저 & 스케줄러 (Optimizer & Scheduler)
# ─────────────────────────────────────────────────────────────────────────────
OPTIMIZER    = 'adamw'           # 'adam' | 'adamw' | 'sgd'
LEARNING_RATE= 1e-4
WEIGHT_DECAY = 1e-4
BETAS        = (0.9, 0.999)

SCHEDULER     = 'cosine'         # 'step' | 'cosine' | 'linear' | None
LR_DECAY_STEPS= [20, 40]         # step 스케줄러일 때
WARMUP_STEPS  = 500
MIN_LR        = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# 5. 학습 설정 (Training Settings)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPUS             = -1            # -1 = 모든 GPU 사용
BATCH_SIZE       = 32
NUM_WORKERS      = 4
EPOCHS           = 50
ACCUMULATE_BATCHES = 1           # gradient accumulation
GRAD_CLIP_VAL    = None          # 1.0 
PRECISION        = 32            # mixed precision (16 or 32)
VAL_CHECK_INTERVAL = 1.0         # 에폭 단위(1.0) 또는 배치 비율(<1.0)

# ─────────────────────────────────────────────────────────────────────────────
# 6. 콜백 설정 (Callbacks)
# ─────────────────────────────────────────────────────────────────────────────
# EarlyStopping
EARLY_STOP_MONITOR  = 'val_pred_error'
EARLY_STOP_MODE     = 'min'
EARLY_STOP_PATIENCE = 5

# ModelCheckpoint
CKPT_MONITOR        = 'val_pred_error'
CKPT_MODE           = 'min'
CKPT_SAVE_TOP_K     = 3         # best k개 저장
CKPT_EVERY_N_EPOCHS = None      # 매 N 에폭마다 저장

CHECKPOINT_DIR      = "../checkpoints"
CHECKPOINT_NAME     = 'xformer-{epoch:02d}-{val_loss:.4f}'

# ─────────────────────────────────────────────────────────────────────────────
# 7. DataLoader 고급 옵션
# ─────────────────────────────────────────────────────────────────────────────
PIN_MEMORY         = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR    = 2