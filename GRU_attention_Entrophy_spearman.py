# rul_model_trainer_all_sets.py (GRU + Attention 기반 RUL 예측)

'''
■ 전체 전략 요약
--------------------------------------------------
- 다채널 진동 데이터의 주파수 도메인 특징을 추출하여 시계열 형태로 모델에 입력
- GRU + Multi-Head Attention 구조로 잔존 수명(RUL) 예측
- 모델 안정화와 작은 RUL 정확도 향상을 위한 log1p 변환 + weighted MAE 손실 함수 적용
- 고장 관련 주파수 및 Spearman 상관 기반 주파수 선택 후 에너지 엔트로피 추출
'''

# ────────────────────────────── 라이브러리 import ──────────────────────────────
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from scipy.stats import kurtosis, skew, spearmanr
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from nptdms import TdmsFile
import matplotlib.pyplot as plt

# ────────────────────────────── 상수 설정 ──────────────────────────────
SELECTED_FREQ_INDICES = {}               # 채널별 선택된 주파수 인덱스 저장
FREQ_VECTOR = None                       # Welch 분석 주파수 벡터
FAULT_FREQS = [140, 93, 78, 6.7]         # 고장 관련 주요 주파수

# ────────────────────────────── ❶ 모델 정의 ──────────────────────────────
def build_gru_attention_model(seq_len, n_feat, gru_units=64, n_heads=4, attn_key_dim=32, ff_dim=32, dropout_rate=0.3):
    '''GRU 기반 시계열 인코더 + Self-Attention → 회귀 출력'''
    x_in = Input(shape=(seq_len, n_feat))
    x = GRU(gru_units, return_sequences=True)(x_in)                         # 시계열 인코딩
    attn = MultiHeadAttention(num_heads=n_heads, key_dim=attn_key_dim, dropout=dropout_rate)(x, x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn)                          # 잔차 연결 + 정규화
    ff = Dense(ff_dim, activation="relu")(x)                               # FFN
    ff = Dense(gru_units)(ff)
    x = LayerNormalization(epsilon=1e-6)(x + ff)
    x = Dropout(dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)                                         # 벡터 풀링
    out = Dense(1)(x)                                                       # 회귀 출력
    return Model(x_in, out, name="GRU_Attention_RUL")

# ────────────────────────────── ❷ 사용자 정의 손실 함수 ──────────────────────────────
def weighted_mae(y_true, y_pred):
    weights = 1 / (K.clip(y_true, 1, np.inf))                               # 작은 RUL에 큰 가중치
    return K.mean(weights * K.abs(y_true - y_pred))

# ────────────────────────────── ❸ 데이터 로드 및 전처리 함수 ──────────────────────────────
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

# ────────────────────────────── ❹ 주파수 선택 함수 ──────────────────────────────
def compute_selected_frequency_indices(file_list, channels, top_n=20):
    '''Spearman 상관 기반으로 상위 주파수 + 고장 주파수 선택'''
    psd_by_channel = {ch: [] for ch in channels}
    rul_list = []

    for file_path, rul in file_list:
        df = load_vibration_data(file_path)
        if df.empty: continue
        for ch in channels:
            if ch not in df.columns: continue
            data = df[ch].values
            f, Pxx = welch(data, fs=25600, nperseg=4096)
            psd_by_channel[ch].append(Pxx)
        rul_list.append(rul)

    selected = {}
    for ch in channels:
        psd_matrix = np.array(psd_by_channel[ch])
        if psd_matrix.shape[0] == 0: continue
        rho_list = [abs(spearmanr(psd_matrix[:, i], rul_list)[0]) for i in range(psd_matrix.shape[1])]
        top_indices = np.argsort(rho_list)[-top_n:]
        fault_indices = [np.argmin(np.abs(f - ff)) for ff in FAULT_FREQS]
        total_indices = sorted(set(top_indices.tolist() + fault_indices))
        selected[ch] = total_indices
        selected_freqs = [f[i] for i in total_indices]
        print(f"{ch} 선택 주파수 (Hz): {selected_freqs}")

    return selected, f

# ────────────────────────────── ❺ 에너지 엔트로피 계산 ──────────────────────────────
def energy_entropy_selected(data, selected_indices):
    f, Pxx = welch(data, fs=25600, nperseg=4096)
    selected = Pxx[selected_indices]
    selected = selected / np.sum(selected)
    selected = selected[selected > 0]
    return -np.sum(selected * np.log(selected))

# ────────────────────────────── ❻ 특징 추출 ──────────────────────────────
def extract_features_from_vibration(vib_df):
    '''기본 통계 + 밴드파워 + 선택 주파수의 에너지 엔트로피'''
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

# ────────────────────────────── ❼ 전체 데이터 처리 ──────────────────────────────
def process_all_sets(top_folder):
    '''모든 세트에서 특징 추출 및 RUL 계산'''
    global SELECTED_FREQ_INDICES, FREQ_VECTOR
    all_rows = []
    rul_pairs = []
    channels = ["CH1", "CH2", "CH3", "CH4"]

    set_folders = sorted([os.path.join(top_folder, d) for d in os.listdir(top_folder) if os.path.isdir(os.path.join(top_folder, d))])

    for set_path in set_folders:
        tdms_files = sorted(glob(os.path.join(set_path, "*.tdms")), key=extract_timestamp)
        if not tdms_files: continue
        file_times = [extract_timestamp(f) for f in tdms_files]
        end_time = max(file_times)
        for file_path, ts in zip(tdms_files, file_times):
            rul = (end_time - ts).total_seconds()
            rul_pairs.append((file_path, rul))

    # 주파수 선택
    SELECTED_FREQ_INDICES, FREQ_VECTOR = compute_selected_frequency_indices(rul_pairs, channels, top_n=20)

    for file_path, rul in rul_pairs:
        try:
            vib_df = load_vibration_data(file_path)
            if vib_df.empty: continue
            features = extract_features_from_vibration(vib_df)
            features['file'] = os.path.basename(file_path)
            features['RUL'] = rul
            all_rows.append(features)
        except Exception as e:
            print(f" 오류 발생: {file_path} - {e}")

    return pd.DataFrame(all_rows)

# ────────────────────────────── ❽ 시계열 시퀀스 생성 ──────────────────────────────
def create_sequences(X, y, window_size=5):
    '''윈도우 기반 시계열 분할'''
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i + window_size - 1])
    return np.array(X_seq), np.array(y_seq)

# ────────────────────────────── ❾ 실행부 (학습 & 평가) ──────────────────────────────
if __name__ == "__main__":
    DATA_ROOT = r"c:/Users/조성찬/OneDrive - UOS/바탕 화면/배어링데이터"
    full_df = process_all_sets(DATA_ROOT)
    full_df = full_df.sort_values(by='file')

    if full_df.empty:
        exit()

    # 스케일링 및 log 변환
    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(full_df.drop(columns=["RUL", "file"]))
    y_all_log = np.log1p(full_df["RUL"].values)    # log1p: log(1 + x)

    X_seq, y_seq = create_sequences(X_all, y_all_log, window_size=5)
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # 모델 정의 및 학습
    model = build_gru_attention_model(seq_len=X_train.shape[1], n_feat=X_train.shape[2])
    model.compile(optimizer=Adam(5e-4), loss=weighted_mae)
    model.fit(X_train, y_train, epochs=5000, batch_size=16, validation_data=(X_val, y_val))

    # 예측 및 후처리
    pred_log = model.predict(X_val)
    y_val_true = np.expm1(y_val)
    pred = np.expm1(pred_log)
    pred = np.clip(pred, 10, None)                     # 음수/너무 작은 RUL 방지

    relative_errors = np.abs((y_val_true - pred.flatten()) / y_val_true)
    relative_errors = relative_errors[np.isfinite(relative_errors)]
    mare = np.mean(relative_errors) * 100
    print(f"\n [전체 기준] 평균 상대 오차 (MARE): {mare:.2f}%")

    model.save("rul_lstm_all_sets.h5")
    print("\n모델이 rul_lstm_all_sets.h5로 저장되었습니다.")