# rul_model_trainer_all_sets.py (with log1p RUL, weighted loss, clipped output)
# rul_model_trainer_all_sets.py (with log1p RUL, weighted loss, clipped output)
'''
# --------------------------------------------------
# ▶ 특징 결합 방식
# 각 채널에서 추출한 특징들을 하나의 벡터로 병렬 결합(concatenate)
# 예시: CH2_mean, CH2_std, CH2_entropy, ..., CH4_band_power, CH4_entropy
#
# ▶ 엔트로피 비중 강화
# 엔트로피 특징에 채널별 가중치 곱셈 (CH2, CH3, CH4 모두 가중치 4)
#
# ▶ Spearman 상관 기반 주파수 선택
# 각 채널에서 상위 20개 + 고장 관련 주파수 포함
#
# ▶ Welch 스펙트럼 분석 설정
# f, Pxx = welch(data, fs=25600, nperseg=4096) : 주파수 분해능 향상
#
# ▶ RUL 예측 개선을 위한 전략 3가지 적용
# 1. RUL log 변환 (log1p) → 모델 안정화
# 2. 손실 함수 weighted MAE → 작은 RUL 강조
# 3. 예측값 클리핑 (min=10) → MARE 튀는 현상 방지
'''
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from scipy.stats import kurtosis, skew, spearmanr
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers  import Adam
from tensorflow.keras import backend as K
from nptdms import TdmsFile
import matplotlib.pyplot as plt

SELECTED_FREQ_INDICES = {}
FREQ_VECTOR = None
FAULT_FREQS = [140, 93, 78, 6.7]
# ──────────────────── ❶ 모델 정의 (GRU + Attention) ────────────────────


def build_gru_attention_model(seq_len, n_feat,
                              gru_units=64,
                              n_heads=4,
                              attn_key_dim=32,
                              ff_dim=32,
                              dropout_rate=0.3):
    """
    seq_len : 윈도우 길이 (TIME dimension)
    n_feat  : 특징 차원
    """
    x_in = Input(shape=(seq_len, n_feat))

    # 1) GRU encoder
    x = GRU(gru_units, return_sequences=True)(x_in)          # (B, T, H)

    # 2) Multi-Head Self-Attention
    attn = MultiHeadAttention(num_heads=n_heads,
                              key_dim=attn_key_dim,
                              dropout=dropout_rate)(x, x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn)           # Residual + LN

    # 3) Position-wise FFN
    ff  = Dense(ff_dim, activation="relu")(x)
    ff  = Dense(gru_units)(ff)
    x   = LayerNormalization(epsilon=1e-6)(x + ff)           # Residual + LN
    x   = Dropout(dropout_rate)(x)

    # 4) 시퀀스 → 벡터 풀링
    x = GlobalAveragePooling1D()(x)

    # 5) 최종 회귀 헤드
    out = Dense(1)(x)

    return Model(x_in, out, name="GRU_Attention_RUL")


# Custom weighted loss
def weighted_mae(y_true, y_pred):
    weights = 1 / (K.clip(y_true, 1, np.inf))
    return K.mean(weights * K.abs(y_true - y_pred))

def load_vibration_data(file_path):
    tdms_file = TdmsFile.read(file_path)
    group_name = tdms_file.groups()[0].name
    vib_channels = tdms_file[group_name].channels()
    vib_data = {ch.name: ch.data for ch in vib_channels}
    return pd.DataFrame(vib_data)

def extract_timestamp(f):
    name = os.path.basename(f)
    time_part = name.split("_")[-1].replace(".tdms", "")
    return pd.to_datetime(time_part, format="%Y%m%d%H%M%S")

def compute_selected_frequency_indices(file_list, channels, top_n=20):
    psd_by_channel = {ch: [] for ch in channels}
    rul_list = []

    for file_path, rul in file_list:
        df = load_vibration_data(file_path)
        if df.empty:
            continue
        for ch in channels:
            if ch not in df.columns:
                continue
            data = df[ch].values
            f, Pxx = welch(data, fs=25600, nperseg=4096)
            psd_by_channel[ch].append(Pxx)
        rul_list.append(rul)

    selected = {}
    for ch in channels:
        psd_matrix = np.array(psd_by_channel[ch])
        if psd_matrix.shape[0] == 0:
            continue
        rho_list = [abs(spearmanr(psd_matrix[:, i], rul_list)[0]) for i in range(psd_matrix.shape[1])]
        top_indices = np.argsort(rho_list)[-top_n:]
        fault_indices = [np.argmin(np.abs(f - ff)) for ff in FAULT_FREQS]
        total_indices = sorted(set(top_indices.tolist() + fault_indices))
        selected[ch] = total_indices
        selected_freqs = [f[i] for i in total_indices]
        print(f"{ch} 선택 주파수 (Hz): {selected_freqs}")

    return selected, f

def energy_entropy_selected(data, selected_indices):
    f, Pxx = welch(data, fs=25600, nperseg=4096)
    selected = Pxx[selected_indices]
    selected = selected / np.sum(selected)
    selected = selected[selected > 0]
    return -np.sum(selected * np.log(selected))

def extract_features_from_vibration(vib_df):
    features = {}
    for ch in vib_df.columns:
        data = vib_df[ch].values
        rms = np.sqrt(np.mean(data**2))
        f, Pxx = welch(data, fs=25600, nperseg=4096)
        features[f'{ch}_mean'] = np.mean(data)
        features[f'{ch}_std'] = np.std(data)
        features[f'{ch}_rms'] = rms
        features[f'{ch}_kurtosis'] = kurtosis(data)
        features[f'{ch}_skew'] = skew(data)
        features[f'{ch}_crest'] = np.max(np.abs(data)) / rms
        features[f'{ch}_band_power'] = np.sum(Pxx)
        ENTROPY_WEIGHTS = {"CH2": 4, "CH3": 4, "CH4": 4}
        if ch in SELECTED_FREQ_INDICES:
            features[f'{ch}_entropy'] = energy_entropy_selected(data, SELECTED_FREQ_INDICES[ch]) ** ENTROPY_WEIGHTS[ch]
    return features

def process_all_sets(top_folder):
    global SELECTED_FREQ_INDICES, FREQ_VECTOR
    all_rows = []
    rul_pairs = []
    channels = ["CH2", "CH3", "CH4"]
    set_folders = sorted([os.path.join(top_folder, d) for d in os.listdir(top_folder) if os.path.isdir(os.path.join(top_folder, d))])

    for set_path in set_folders:
        tdms_files = sorted(glob(os.path.join(set_path, "*.tdms")), key=extract_timestamp)
        if not tdms_files:
            continue
        file_times = [extract_timestamp(f) for f in tdms_files]
        end_time = max(file_times)
        for file_path, ts in zip(tdms_files, file_times):
            rul = (end_time - ts).total_seconds()
            rul_pairs.append((file_path, rul))

    SELECTED_FREQ_INDICES, FREQ_VECTOR = compute_selected_frequency_indices(rul_pairs, channels, top_n=20)

    for file_path, rul in rul_pairs:
        try:
            vib_df = load_vibration_data(file_path)
            if vib_df.empty:
                continue
            features = extract_features_from_vibration(vib_df)
            features['file'] = os.path.basename(file_path)
            features['RUL'] = rul
            all_rows.append(features)
        except Exception as e:
            print(f" 오류 발생: {file_path} - {e}")

    return pd.DataFrame(all_rows)

def create_sequences(X, y, window_size=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i + window_size - 1])
    return np.array(X_seq), np.array(y_seq)

if __name__ == "__main__":
    DATA_ROOT = r"c:/Users/조성찬/OneDrive - UOS/바탕 화면/배어링데이터"
    full_df = process_all_sets(DATA_ROOT)
    full_df = full_df.sort_values(by='file')

    if full_df.empty:
        exit()

    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(full_df.drop(columns=["RUL", "file"]))
    y_all_log = np.log1p(full_df["RUL"].values)  # log1p 변환

    X_seq, y_seq = create_sequences(X_all, y_all_log, window_size=5)
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # ──────────────────── ❷ 모델 빌드 & 학습 부분 교체 ────────────────────
    # 기존 LSTM 모델 코드 대신 ↓↓↓ 사용
    model = build_gru_attention_model(
                seq_len   = X_train.shape[1],
                n_feat    = X_train.shape[2],
                gru_units = 64,     # 필요하면 조정
                n_heads   = 4,
                attn_key_dim = 32,
                ff_dim    = 32,
                dropout_rate = 0.3
    )

    model.compile(optimizer=Adam(5e-4), loss=weighted_mae)

    model.fit(
        X_train, y_train,
        epochs=5000,
        batch_size=16,
        validation_data=(X_val, y_val)
    )

    pred_log = model.predict(X_val)
    # 예측값 역변환 및 클리핑
    # 역변환
    y_val_true = np.expm1(y_val)
    pred = np.expm1(pred_log)

    # 예측 클리핑은 유지 (모델이 마이너스 RUL 예측할 수도 있으니까)
    pred = np.clip(pred, 10, None)

    # 진짜 전체에 대해 상대 오차 계산
    relative_errors = np.abs((y_val_true - pred.flatten()) / y_val_true)

    # 무한대 또는 NaN 제거 (계산은 하되 평균에 포함 안 함)
    relative_errors = relative_errors[np.isfinite(relative_errors)]

    mare = np.mean(relative_errors) * 100
    print(f"\n [전체 기준] 평균 상대 오차 (MARE): {mare:.2f}%")


    model.save("rul_lstm_all_sets.h5")
    print("\n모델이 rul_lstm_all_sets.h5로 저장되었습니다.")
