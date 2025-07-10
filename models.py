import torch
from torch.utils.data import DataLoader
from dataset import VesselPatchDataset
from context_encoder import ContextEncoder
from fusion_encoder import FusionEncoder
from traj_encoder_4hot import FourHotTrajectoryEncoder

# --- Hyperparameters and Configuration ---
LAT_SIZE = 204       # number of latitude bins (example)
LON_SIZE = 320       # number of longitude bins (example)
SOG_SIZE = 30        # number of speed-over-ground bins (example)
COG_SIZE = 36        # number of course-over-ground bins (example)

N_LAT_EMBD = 32      # embedding dim per latitude bin
N_LON_EMBD = 32      # embedding dim per longitude bin
N_SOG_EMBD = 16      # embedding dim per SOG bin
N_COG_EMBD = 16      # embedding dim per COG bin

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

# --- Dataset and DataLoader ---
# Assumes VesselPatchDataset returns 'trajectory': (L,4) and 'context': (3,H,W)
dataset = VesselPatchDataset(
    ais_data_file='path/to/ais_data.pkl',
    seq_len=MAX_LEN, # need to fix 
    max_seqlen=MAX_LEN,
    patch_radius_km=5.0,
    time_step_mins=10,
)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

# --- Model Initialization ---
# Context encoder outputs fixed D_MODEL embedding
context_encoder = ContextEncoder(in_channels=3, out_dim=D_MODEL) # 3channel : (location, sog, cog)
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
# Fusion encoder merges (B,L,D_MODEL) with (B,1,D_MODEL)
fusion_encoder = FusionEncoder(
    d_model=D_MODEL,
    n_head=N_HEAD,
    dim_feedforward=DIM_FF,
    dropout=DROPOUT,
    num_layers=2
)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
context_encoder.to(device)
traj_encoder.to(device)
fusion_encoder.to(device)

# --- Forward Pass Example ---
for batch in dataloader:
    # batch['trajectory']: (B, L, 4)
    # batch['context']:    (B, 3, H, W)
    traj_seq = batch['trajectory'].to(device)    # (B, L, 4)
    context_map = batch['context'].to(device)    # (B, 3, H, W)

    # Encode
    context_emb = context_encoder(context_map)                # (B, D_MODEL)
    traj_emb = traj_encoder(traj_seq)                         # (B, L, D_MODEL)
    fused_emb = fusion_encoder(traj_emb, context_emb)         # (B, L, D_MODEL)

    # fused_emb can now be fed into your decoder
    print(f"Fused embedding shape: {fused_emb.shape}")
    break