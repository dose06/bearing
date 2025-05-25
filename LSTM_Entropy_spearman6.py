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
# -> 이 후 데이터 셋의 마지막 파일 제외하므로써 예측값 클리핑 제외
'''
import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import kurtosis, skew, spearmanr
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import backend as K
from nptdms import TdmsFile
import matplotlib.pyplot as plt

SELECTED_FREQ_INDICES = {}
FREQ_VECTOR = None
FAULT_FREQS = [140, 93, 78, 6.7]

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
        ENTROPY_WEIGHTS = {"CH1":4, "CH2": 4, "CH3": 4, "CH4": 4}
        if ch in SELECTED_FREQ_INDICES:
            features[f'{ch}_entropy'] = energy_entropy_selected(data, SELECTED_FREQ_INDICES[ch]) ** ENTROPY_WEIGHTS[ch]
    return features

def process_all_sets(top_folder):
    global SELECTED_FREQ_INDICES, FREQ_VECTOR
    all_rows = []
    rul_pairs = []
    channels = ["CH1", "CH2", "CH3", "CH4"]
    set_folders = sorted([os.path.join(top_folder, d) for d in os.listdir(top_folder) if os.path.isdir(os.path.join(top_folder, d))])

    for set_path in set_folders:
        tdms_files = sorted(glob(os.path.join(set_path, "*.tdms")), key=extract_timestamp)
        if not tdms_files:
            continue
        file_times = [extract_timestamp(f) for f in tdms_files]
        end_time = max(file_times)

        for file_path, ts in zip(tdms_files[:-1], file_times[:-1]):     # ← 마지막 파일 제외file_times[:-1]):
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


def compute_percent_error(actual, predicted):
    """예측 오차 (백분율 %) 계산"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    nonzero_mask = actual != 0
    eri = np.zeros_like(actual)
    eri[nonzero_mask] = 100 * (actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask]
    return eri

def compute_arul_score(eri):
    """ERI (백분율 오차)를 기반으로 A_RUL 점수 계산"""
    eri = np.array(eri)
    score = np.where(
        eri <= 0,
        np.exp(-np.log(0.5) * eri / 20),
        np.exp(+np.log(0.5) * eri / 50)
    )
    return score


WINDOW = 5                      # 시퀀스 길이
BINS   = [-1, 30, 300, 2000, np.inf]   # stratify 구간
# ─────────────────────────────── main ────────────────────────────────
if __name__ == "__main__":
    DATA_ROOT = r"c:/Users/조성찬/OneDrive - UOS/바탕 화면/배어링데이터"
    full_df   = process_all_sets(DATA_ROOT).sort_values('file')

    if full_df.empty: raise SystemExit("❌ full_df가 비어 있습니다")

    # ──★ stratify 라벨 (RUL 구간)
    labels = pd.cut(full_df["RUL"], bins=BINS, labels=False)

    # 스케일링
    scaler = MinMaxScaler()
    X_all  = scaler.fit_transform(full_df.drop(columns=["RUL","file"]))
    y_all  = np.log1p(full_df["RUL"].values)           # log1p 변환

    # 시퀀스
    X_seq, y_seq       = create_sequences(X_all, y_all, window_size=WINDOW)
    labels_seq         = labels[WINDOW-1:].to_numpy()  # ──★ 앞 WINDOW-1개 drop

    # stratified split (train 90 % / val 10 %)
    X_tr, X_val, y_tr, y_val, lab_tr, lab_val = train_test_split(
        X_seq, y_seq, labels_seq,
        test_size=0.1, random_state=42, stratify=labels_seq)

    # ──────────────────── LSTM ────────────────────
    model = Sequential([
        LSTM(64, input_shape=(WINDOW, X_seq.shape[2])),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=weighted_mae)
    model.fit(X_tr, y_tr, epochs=5000, batch_size=16,
              validation_data=(X_val, y_val))

    # ───────── 평가 (역변환, 클리핑, MARE / A_RUL) ─────────
    pred_log = model.predict(X_val).flatten()
    pred     = np.clip(np.expm1(pred_log), 10, None)
    y_true   = np.expm1(y_val)

    eri   = compute_percent_error(y_true, pred)
    a_rul = compute_arul_score(eri)

    print(f"\n평균 상대 오차 (MARE)   : {np.mean(np.abs(eri)):.2f}%")
    print(f"A_RUL 평균 점수        : {np.mean(a_rul):.4f}")

    model.save("rul_lstm_all_sets.h5")
    print("✅ 모델 저장 완료")