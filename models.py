import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VesselPatchDataset
from context_encoder import ContextEncoder
from fusion_encoder import FusionEncoder
from traj_encoder_4hot import FourHotTrajectoryEncoder
from decoder import AISDecoder 

# --- Hyperparameters and Configuration ---
LAT_SIZE = 204       # number of latitude bins (example)
LON_SIZE = 320       # number of longitude bins (example)
SOG_SIZE = 30        # number of speed-over-ground bins (example)
COG_SIZE = 36        # number of course-over-ground bins (example)

N_LAT_EMBD = 256      # embedding dim per latitude bin
N_LON_EMBD = 256     # embedding dim per longitude bin
N_SOG_EMBD = 128      # embedding dim per SOG bin
N_COG_EMBD = 128      # embedding dim per COG bin

# combined model dimension
D_MODEL = N_LAT_EMBD + N_LON_EMBD + N_SOG_EMBD + N_COG_EMBD

N_HEAD = 8           # number of attention heads
NUM_LAYERS = 3       # number of fusion & traj encoder layers
DIM_FF = 512         # feedforward dimension
DROPOUT = 0.1        # dropout probability
MAX_LEN = 120       # maximum trajectory sequence length
init_seqlen = 18
min_seqlen = 36

BATCH_SIZE = 32
NUM_WORKERS = 4

path = "/home/iscilab/AIS/TrAISformer/data/ct_dma/ct_dma_test.pkl"

# --- Dataset and DataLoader ---
# Assumes VesselPatchDataset returns 'trajectory': (L,4) and 'context': (3,H,W)
dataset = VesselPatchDataset(
    ais_data_file=path,
    seq_len=MAX_LEN, # need to fix 
    max_seqlen=MAX_LEN,
    patch_radius_km=5.0,
    time_step_mins=10,
)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True 
)

# --- Model Initialization ---
# Context encoder outputs fixed D_MODEL embedding
context_encoder = ContextEncoder(in_channels=3, out_dim=D_MODEL) # 3channel : (location, sog, cog)
# (B, D_MODEL)

# Four-hot trajectory encoder
traj_encoder = FourHotTrajectoryEncoder(
    lat_size=LAT_SIZE,
    lon_size=LON_SIZE,
    sog_size=SOG_SIZE,
    cog_size=COG_SIZE,
    n_lat_embd=N_LAT_EMBD,
    n_lon_embd=N_LON_EMBD,
    n_sog_embd=N_SOG_EMBD,
    n_cog_embd=N_COG_EMBD,
    n_head=N_HEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FF,
    dropout=DROPOUT,
    max_len=MAX_LEN
)
# (B, L, D_MODEL)

# Fusion encoder merges (B,L,D_MODEL) with (B,1,D_MODEL)
fusion_encoder = FusionEncoder(
    d_model=D_MODEL,
    n_head=N_HEAD,
    dim_feedforward=DIM_FF,
    dropout=DROPOUT,
    num_layers=2
) # (B, L, D_MODEL) 

# Decoder
decoder = AISDecoder(
    d_model=D_MODEL, n_head=N_HEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FF,
    dropout=DROPOUT, max_len=MAX_LEN, lat_size=LAT_SIZE, lon_size=LON_SIZE,
    sog_size=SOG_SIZE, cog_size=COG_SIZE, lat_embd=N_LAT_EMBD, lon_embd=N_LON_EMBD,
    sog_embd=N_SOG_EMBD, cog_embd=N_COG_EMBD
)
# Loss Function
loss_fn = nn.CrossEntropyLoss()

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
context_encoder.to(device)
traj_encoder.to(device)
fusion_encoder.to(device)
decoder.to(device)

# --- Forward Pass Example ---
for batch in dataloader:
    # batch['trajectory']: (B, MAX_LEN, 4)
    # batch['mask']:       (B, MAX_LEN)
    # batch['context']:    (B, 3, H, W)
    traj_seq   = batch['trajectory'].to(device)          # (B, L, 4)
    mask       = batch['mask'].to(device)                # (B, L)
    context_map= batch['context'].to(device)             # (B, 3, H, W)

    # Encode context
    context_emb = context_encoder(context_map)           # (B, D_MODEL)

    # Encode trajectory with padding mask
    src_key_padding_mask = ~mask.bool()                  # True=pad positions
    traj_emb = traj_encoder(
        traj_seq,
        src_key_padding_mask=src_key_padding_mask
    )                                                       # (B, L, D_MODEL)
    
    fused_emb = fusion_encoder(traj_emb, context_emb)         # (B, L, D_MODEL)

    # fused_emb can now be fed into your decoder
    print(f"Fused embedding shape: {fused_emb.shape}")
    
    # --- 2. 디코딩 및 손실 계산 ---
    # 디코더 입력 및 정답(label) 준비
    # 입력: <SOS> 토큰으로 시작하고 마지막 시점 제외 (오른쪽으로 shift)
    # 정답: 첫 시점 제외하고 <EOS> 토큰으로 끝남
    decoder_input = traj_seq # 디코더 입력 (실제로는 <SOS> 토큰 등이 필요)

    # 디코더 순전파
    logits, loss = decoder(
        tgt_seq=decoder_input,
        memory=fused_emb,
        tgt_key_padding_mask=src_key_padding_mask,
        memory_key_padding_mask=src_key_padding_mask,
    )

    print(f"Lat Logits shape: {logits[0].shape}")
    print(f"Total Loss: {loss.item()}")
    
    # 역전파 및 파라미터 업데이트 (실제 학습 시)
    # total_loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()
    
    break # 테스트를 위해 첫 배치만 실행