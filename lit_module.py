import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from dataset import VesselPatchDataset
from context_encoder import ContextEncoder
from fusion_encoder import FusionEncoder
from traj_encoder_4hot import FourHotTrajectoryEncoder
from decoder import AISDecoder
from utils import haversine
import config 

class TrajDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data_path   = config.TRAIN_DATA_PATH
        self.valid_path   = config.VAL_DATA_PATH
        self.test_path    = config.TEST_DATA_PATH
        self.seq_len      = config.MAX_LEN
        self.max_seqlen   = config.MAX_LEN
        self.batch_size   = config.BATCH_SIZE
        self.num_workers  = config.NUM_WORKERS
        self.patch_radius_km = config.PATCH_RADIUS_KM
        self.time_step_mins    = config.TIME_STEP_MINS
        
    def setup(self, stage=None):
        # train/val/test 분할 대신 하나의 경로에서 모두 불러오는 예시
        self.train_dataset = VesselPatchDataset(
            ais_data_file=self.data_path,
            seq_len=self.seq_len,
            max_seqlen=self.max_seqlen,
            patch_radius_km=self.patch_radius_km,
            time_step_mins=self.time_step_mins
        )
        self.valid_dataset = VesselPatchDataset(
            ais_data_file=self.valid_path,
            seq_len=self.seq_len,
            max_seqlen=self.max_seqlen,
            patch_radius_km=self.patch_radius_km,
            time_step_mins=self.time_step_mins
        )
        self.test_dataset = VesselPatchDataset(
            ais_data_file=self.test_path,
            seq_len=self.seq_len,
            max_seqlen=self.max_seqlen,
            patch_radius_km=self.patch_radius_km,
            time_step_mins=self.time_step_mins
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=config.PIN_MEMORY,
                          persistent_workers=config.PERSISTENT_WORKERS, prefetch_factor=config.PREFETCH_FACTOR)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=config.PIN_MEMORY,
                          persistent_workers=config.PERSISTENT_WORKERS, prefetch_factor=config.PREFETCH_FACTOR)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=config.PIN_MEMORY,
                          persistent_workers=config.PERSISTENT_WORKERS, prefetch_factor=config.PREFETCH_FACTOR)

class XFormerLit(pl.LightningModule):
    def __init__(self,
                 lat_size=config.LAT_SIZE,    
                 lon_size=config.LON_SIZE,
                 sog_size=config.SOG_SIZE,
                 cog_size=config.COG_SIZE,
                 n_lat_embd=config.N_LAT_EMBD,
                 n_lon_embd=config.N_LON_EMBD,
                 n_sog_embd=config.N_SOG_EMBD,
                 n_cog_embd=config.N_COG_EMBD,
                 d_model=config.D_MODEL,
                 n_head=config.N_HEAD,
                 num_layers_enc=config.NUM_LAYERS_ENC,
                 num_layers_fus=config.NUM_LAYERS_FUS,
                 num_layers_dec=config.NUM_LAYERS_DEC,
                 dim_feedforward=config.DIM_FF,
                 dropout=config.DROPOUT,
                 max_len=config.MAX_LEN,
                 lr=config.LEARNING_RATE):
        super().__init__()
        # save hyperparameters 자동 저장
        self.save_hyperparameters()
        # 모델 구성
        self.context_encoder = ContextEncoder(in_channels=config.CONTEXT_CHANNEL, out_dim=d_model)
        self.traj_encoder = FourHotTrajectoryEncoder(
            lat_size, lon_size, sog_size, cog_size,
            n_lat_embd, n_lon_embd, n_sog_embd, n_cog_embd,
            n_head, num_layers_enc, dim_feedforward, dropout, max_len
        )
        self.fusion_encoder = FusionEncoder(
            d_model, n_head, dim_feedforward, dropout, num_layers_fus
        )
        self.decoder = AISDecoder(
            d_model, n_head, num_layers_dec, dim_feedforward, dropout,
            max_len, lat_size, lon_size, sog_size, cog_size,
            n_lat_embd, n_lon_embd, n_sog_embd, n_cog_embd
        )

    def forward(self, traj_seq, context_map, mask):
        if torch.isnan(traj_seq).any() or torch.isnan(context_map).any():
            print("!!! NaN found in model input !!!")
        # 1) encode
        context_emb = self.context_encoder(context_map)
        if torch.isnan(context_emb).any():
            print("!!! NaN occurred after ContextEncoder !!!")
        src_key_padding_mask = ~mask.bool()
        traj_emb = self.traj_encoder(traj_seq, src_key_padding_mask)
        if torch.isnan(traj_emb).any():
            print("!!! NaN occurred after TrajectoryEncoder !!!")
        fused = self.fusion_encoder(traj_emb, context_emb)
        if torch.isnan(fused).any():
            print("!!! NaN occurred after FusionEncoder !!!")
        # 2) decode
        logits, loss = self.decoder(traj_seq, fused, src_key_padding_mask)
        return logits, loss
    
    def _compute_pred_error(self, traj_seq, logits, mask):
        # 1) argmax → normalized → degree 복원
        lat_logits, lon_logits, sog_logits, cog_logits = logits
        lat_idx = lat_logits.argmax(dim=-1)
        lon_idx = lon_logits.argmax(dim=-1)
        lat_n = lat_idx.float() / (self.hparams.lat_size - 1)
        lon_n = lon_idx.float() / (self.hparams.lon_size - 1)
        lat_deg = lat_n * (config.LAT_MAX - config.LAT_MIN) + config.LAT_MIN
        lon_deg = lon_n * (config.LON_MAX - config.LON_MIN) + config.LON_MIN

        # 2) 실제 위·경도 degree
        true_lat_n = traj_seq[..., 0]
        true_lon_n = traj_seq[..., 1]
        true_lat = true_lat_n * (config.LAT_MAX - config.LAT_MIN) + config.LAT_MIN
        true_lon = true_lon_n * (config.LON_MAX - config.LON_MIN) + config.LON_MIN

        # 3) degree → radian 변환
        lat_pred_rad = torch.deg2rad(lat_deg)
        lon_pred_rad = torch.deg2rad(lon_deg)
        lat_true_rad = torch.deg2rad(true_lat)
        lon_true_rad = torch.deg2rad(true_lon)

        # 4) coords 형태로 스택: (..., 2)
        #    index 0 은 latitude, 1 은 longitude
        pred_coords = torch.stack([lat_pred_rad, lon_pred_rad], dim=-1)
        true_coords = torch.stack([lat_true_rad, lon_true_rad], dim=-1)

        # 5) mask 적용 전	Boolean mask
        mask_bool = mask.bool()

        # 6) 새로운 haversine 함수 호출
        dists = haversine(true_coords, pred_coords)
        dists = dists[mask_bool]

        # 7) 평균 오차
        return dists.mean()

    def training_step(self, batch, batch_idx):
        traj_seq = batch['trajectory']
        mask     = batch['mask']
        context  = batch['context']
        _, loss = self.forward(traj_seq, context, mask)
        self.log('train_loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        traj_seq = batch['trajectory']
        mask     = batch['mask']
        context  = batch['context']
        logits, loss = self.forward(traj_seq, context, mask)
        self.log('val_loss', loss, prog_bar=True)
        pred_err = self._compute_pred_error(traj_seq, logits, mask)
        self.log('val_pred_error', pred_err, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        traj_seq = batch['trajectory']
        mask     = batch['mask']
        context  = batch['context']
        logits, _ = self.forward(traj_seq, context, mask)
        test_err = self._compute_pred_error(traj_seq, logits, mask)
        self.log('test_pred_error', test_err, prog_bar=True)

        

    def configure_optimizers(self):
        # optimizer choice
        if config.OPTIMIZER == 'adamw':
            opt = torch.optim.AdamW(self.parameters(),
                                    lr=self.hparams.lr,
                                    weight_decay=config.WEIGHT_DECAY,
                                    betas=config.BETAS)
        # scheduler (optional)
        if config.SCHEDULER == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=config.EPOCHS,
                eta_min=config.MIN_LR
            )
            return [opt], [ {'scheduler': scheduler, 'interval':'epoch'} ]
        return opt

# === Trainer setting  ===
early_stop = EarlyStopping(
    monitor=config.EARLY_STOP_MONITOR,
    patience=config.EARLY_STOP_PATIENCE,
    mode=config.EARLY_STOP_MODE,
)
checkpoint = ModelCheckpoint(
    monitor=config.CKPT_MONITOR,
    mode=config.CKPT_MODE,
    save_top_k=config.CKPT_SAVE_TOP_K,
    filename=config.CHECKPOINT_NAME,
    dirpath=config.CHECKPOINT_DIR,
    every_n_epochs=config.CKPT_EVERY_N_EPOCHS 
)

def logger(logger): 
    if logger == 'wandb':
        wandb_logger = pl.loggers.WandbLogger(
        project=config.PROJECT_NAME,
        name=config.RUN_NAME
    )
        return wandb_logger
    elif logger == 'tensorboard':
        tb_logger = pl.loggers.TensorBoardLogger(
            save_dir="tb_logs",
            name=config.RUN_NAME
        )
        return tb_logger
    else:
        return False 


trainer = pl.Trainer(
    logger=logger(config.LOGGER), 
    accelerator='gpu', 
    devices=config.GPUS,
    max_epochs=config.EPOCHS,             
    precision=config.PRECISION,           
    accumulate_grad_batches=config.ACCUMULATE_BATCHES,  
    gradient_clip_val=config.GRAD_CLIP_VAL,            
    val_check_interval=config.VAL_CHECK_INTERVAL,      
    callbacks=[checkpoint],
    default_root_dir=config.CHECKPOINT_DIR
)

# Training 
pl.seed_everything(config.SEED, workers=True) 

dm = TrajDataModule() 
model = XFormerLit()
trainer.fit(model, dm)
# end train and evaluate 
trainer.test(model, datamodule=dm)