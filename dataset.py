import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from datetime import datetime, timedelta, timezone

# --- 상수 & 헬퍼 ---
LAT_MIN, LAT_MAX = 55.5, 58.0
LON_MIN, LON_MAX = 10.3, 13.0
PATCH_SIZE = 64

def normalize_lat(lat):   return (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
def normalize_lon(lon):   return (lon - LON_MIN) / (LON_MAX - LON_MIN)
def denormalize_lat(n):   return n * (LAT_MAX - LAT_MIN) + LAT_MIN
def denormalize_lon(n):   return n * (LON_MAX - LON_MIN) + LON_MIN

class VesselPatchDataset(Dataset):
    def __init__(self,
                 ais_data_file: str,
                 seq_len: int,
                 max_seqlen: int = 96,
                 patch_radius_km: float = 5.0,
                 time_step_mins: int = 10):
        # 1) 전체 AIS 리스트 로드
        with open(ais_data_file, 'rb') as f:
            self.ais_data = pickle.load(f)
        # 2) 모든 (vessel_idx, window_start) 쌍을 미리 모아 둡니다
        self.samples = []
        for v_idx, V in enumerate(self.ais_data):
            traj = V['traj']  # (T,5)
            T = len(traj)
            Nw = max(0, T - (seq_len - 1))
            for start in range(Nw):
                self.samples.append((v_idx, start))
        # 3) 나머지
        self.seq_len = seq_len
        self.max_seqlen = max_seqlen
        self.time_step = timedelta(minutes=time_step_mins)
        self.patch_km = patch_radius_km

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        v_idx, start = self.samples[idx]
        V = self.ais_data[v_idx]
        traj = V['traj']  # shape (T,5)
        times = traj[:,4].astype(int)
        # 1) 시퀀스 끝 시점의 timestamp
        end_ts = times[start + self.seq_len - 1]
        end_time = datetime.fromtimestamp(int(end_ts), tz=timezone.utc)

        # 2) raw seq 생성
        raw = np.zeros((self.seq_len, 4), dtype=np.float32)
        for j in range(self.seq_len):
            target_time = end_time - self.time_step * (self.seq_len - 1 - j)
            ts = int(target_time.timestamp())
            i = np.abs(times - ts).argmin()
            raw[j] = traj[i,:4]  # lat,lon,sog,cog

        # 3) pad/trim → fixed max_seqlen
        seqlen = min(self.seq_len, self.max_seqlen)
        seq = np.zeros((self.max_seqlen, 4), dtype=np.float32)
        seq[:seqlen] = raw[:seqlen]

        # 4) mask
        mask = np.zeros(self.max_seqlen, dtype=np.float32)
        mask[:seqlen] = 1.0

        # 5) context rasterization (기존 로직 그대로)
        #    - end_time에 맞춰 모든 vessel에 대해 패치 생성
        #    - center는 this vessel의 위치
        ts_key = int(end_time.timestamp())
        # find center lat/lon normalized
        lat_cn = raw[-1,0]; lon_cn = raw[-1,1]
        # compute patch bounds…
        lat_c_deg = denormalize_lat(lat_cn)
        dlat = self.patch_km/111.0; dlon = self.patch_km/(111.0*np.cos(np.deg2rad(lat_c_deg)))
        dlat_n = dlat/(LAT_MAX-LAT_MIN); dlon_n = dlon/(LON_MAX-LON_MIN)
        lat_min, lat_max = lat_cn-dlat_n, lat_cn+dlat_n
        lon_min, lon_max = lon_cn-dlon_n, lon_cn+dlon_n
        img = np.zeros((3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        for other in self.ais_data:
            oth_traj = other['traj']; oth_times = oth_traj[:,4].astype(int)
            j = np.abs(oth_times - ts_key).argmin()
            lat_o, lon_o = oth_traj[j,0], oth_traj[j,1]
            if not (lat_min<=lat_o<=lat_max and lon_min<=lon_o<=lon_max):
                continue
            y = (lat_o-lat_min)/(lat_max-lat_min)
            x = (lon_o-lon_min)/(lon_max-lon_min)
            row = int((1-y)*(PATCH_SIZE-1)); col = int(x*(PATCH_SIZE-1))
            if other['mmsi']==V['mmsi']:
                img[0, row, col] = 1.0 
                img[1, row, col] = oth_traj[j, 2]
                img[2, row, col] = oth_traj[j, 3]
            else : 
                img[0, row, col] = 2.0
                img[1, row, col] = oth_traj[j, 2]
                img[2, row, col] = oth_traj[j, 3]

        return {
            'trajectory': torch.from_numpy(seq),    # (max_seqlen,4)
            'mask':       torch.from_numpy(mask),   # (max_seqlen,)
            'context':    torch.from_numpy(img),    # (3,H,W)
        }