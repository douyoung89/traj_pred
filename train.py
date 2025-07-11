import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

# 제공된 다른 .py 파일에서 필요한 클래스와 변수들을 불러옵니다.
from dataset import VesselPatchDataset
from context_encoder import ContextEncoder
from fusion_encoder import FusionEncoder
from traj_encoder_4hot import FourHotTrajectoryEncoder
from decoder import AISDecoder
import config 

# --- Hyperparameters and Configuration (models.py에서 가져옴) ---
# 데이터셋 및 모델의 각종 크기 설정
LAT_SIZE = config.LAT_SIZE
LON_SIZE = config.LON_SIZE
SOG_SIZE = config.SOG_SIZE
COG_SIZE = config.COG_SIZE

# 임베딩 차원 설정
N_LAT_EMBD = config.N_LAT_EMBD
N_LON_EMBD = config.N_LON_EMBD
N_SOG_EMBD = config.N_SOG_EMBD
N_COG_EMBD = config.N_COG_EMBD

# D_MODEL: 모든 임베딩 차원의 합
D_MODEL = N_LAT_EMBD + N_LON_EMBD + N_SOG_EMBD + N_COG_EMBD

# 트랜스포머 관련 하이퍼파라미터
N_HEAD = config.N_HEAD
NUM_LAYERS_FUS = config.NUM_LAYERS_FUS
NUM_LAYERS_DEC = config.NUM_LAYERS_DEC
NUM_LAYERS_ENC = config.NUM_LAYERS_ENC
DIM_FF = config.DIM_FF
DROPOUT = config.DROPOUT
MAX_LEN = config.MAX_LEN

# 학습 관련 하이퍼파라미터
BATCH_SIZE = config.BATCH_SIZE
NUM_WORKERS = config.NUM_WORKERS
LEARNING_RATE = config.LEARNING_RATE
EPOCHS = config.EPOCHS

# 데이터 경로 및 체크포인트 저장 경로 설정
DATA_PATH = config.TRAIN_DATA_PATH # 학습용 데이터 경로
VAL_DATA_PATH = config.VAL_DATA_PATH 
TEST_DATA_PATH = config.TEST_DATA_PATH
CHECKPOINT_DIR = config.CHECKPOINT_DIR

def main():
    # --- 장치 설정 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 체크포인트 저장 디렉토리 생성
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- 데이터셋 및 데이터로더 ---
    train_dataset = VesselPatchDataset(
        ais_data_file=DATA_PATH,
        seq_len=MAX_LEN,
        max_seqlen=MAX_LEN,
    )
    val_dataset = VesselPatchDataset(
        ais_data_file=VAL_DATA_PATH,
        seq_len=MAX_LEN,
        max_seqlen=MAX_LEN,
    )
    test_dataset = VesselPatchDataset(
        ais_data_file=TEST_DATA_PATH, 
        seq_len=MAX_LEN,
        max_seqlen=MAX_LEN,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # model 
    context_encoder = ContextEncoder(in_channels=3, out_dim=D_MODEL).to(device)
    traj_encoder = FourHotTrajectoryEncoder(
        lat_size=LAT_SIZE, lon_size=LON_SIZE, sog_size=SOG_SIZE, cog_size=COG_SIZE,
        n_lat_embd=N_LAT_EMBD, n_lon_embd=N_LON_EMBD, n_sog_embd=N_SOG_EMBD, n_cog_embd=N_COG_EMBD,
        n_head=N_HEAD, num_layers=NUM_LAYERS_ENC, dim_feedforward=DIM_FF, dropout=DROPOUT, max_len=MAX_LEN
    ).to(device)
    fusion_encoder = FusionEncoder(
        d_model=D_MODEL, n_head=N_HEAD, dim_feedforward=DIM_FF, dropout=DROPOUT, num_layers=NUM_LAYERS_FUS
    ).to(device)
    decoder = AISDecoder(
        d_model=D_MODEL, n_head=N_HEAD, num_layers=NUM_LAYERS_DEC, dim_feedforward=DIM_FF,
        dropout=DROPOUT, max_len=MAX_LEN, lat_size=LAT_SIZE, lon_size=LON_SIZE,
        sog_size=SOG_SIZE, cog_size=COG_SIZE, lat_embd=N_LAT_EMBD, lon_embd=N_LON_EMBD,
        sog_embd=N_SOG_EMBD, cog_embd=N_COG_EMBD
    ).to(device)

    # optimizer 
    params = (
        list(context_encoder.parameters()) +
        list(traj_encoder.parameters()) +
        list(fusion_encoder.parameters()) +
        list(decoder.parameters())
    )
    optimizer = optim.Adam(params, lr=LEARNING_RATE)

    # --- 학습 및 검증 루프 ---
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # --- 학습(Training) ---
        context_encoder.train()
        traj_encoder.train()
        fusion_encoder.train()
        decoder.train()

        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for batch in train_pbar:
            # 데이터를 장치로 이동
            traj_seq = batch['trajectory'].to(device)
            mask = batch['mask'].to(device)
            context_map = batch['context'].to(device)

            # 순전파 (Forward Pass)
            optimizer.zero_grad()

            context_emb = context_encoder(context_map)
            src_key_padding_mask = ~mask.bool()
            traj_emb = traj_encoder(traj_seq, src_key_padding_mask=src_key_padding_mask)
            fused_emb = fusion_encoder(traj_emb, context_emb)

            # 디코더 입력 준비 (실제 구현에서는 <SOS> 토큰 등을 고려해야 함)
            decoder_input = traj_seq

            _, loss = decoder(
                tgt_seq=decoder_input,
                memory=fused_emb,
                tgt_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            # 역전파 (Backward Pass)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")

        # --- 검증(Validation) ---
        context_encoder.eval()
        traj_encoder.eval()
        fusion_encoder.eval()
        decoder.eval()

        total_val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")

        with torch.no_grad():
            for batch in val_pbar:
                traj_seq = batch['trajectory'].to(device)
                mask = batch['mask'].to(device)
                context_map = batch['context'].to(device)

                context_emb = context_encoder(context_map)
                src_key_padding_mask = ~mask.bool()
                traj_emb = traj_encoder(traj_seq, src_key_padding_mask=src_key_padding_mask)
                fused_emb = fusion_encoder(traj_emb, context_emb)
                decoder_input = traj_seq

                _, loss = decoder(
                    tgt_seq=decoder_input,
                    memory=fused_emb,
                    tgt_key_padding_mask=src_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                total_val_loss += loss.item()
                val_pbar.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f}")

        # --- 모델 저장 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'context_encoder_state_dict': context_encoder.state_dict(),
                'traj_encoder_state_dict': traj_encoder.state_dict(),
                'fusion_encoder_state_dict': fusion_encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

if __name__ == '__main__':
    main()