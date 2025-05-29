import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import kurtosis, skew, spearmanr
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential,  load_model
from tensorflow.keras.layers import LSTM, Dense
from nptdms import TdmsFile


# ▶ Spearman 기반 선택된 주파수 인덱스 초기화
SELECTED_FREQ_INDICES = {}
FREQ_VECTOR = None

# ▶ 진동 데이터 불러오기
def load_vibration_data(file_path):
    tdms_file = TdmsFile.read(file_path)
    group_name = tdms_file.groups()[0].name  # 진동 데이터가 들어있는 첫 그룹
    vib_channels = tdms_file[group_name].channels()
    vib_data = {ch.name: ch.data for ch in vib_channels}
    return pd.DataFrame(vib_data)

# ▶ 시간 정보 추출 함수
def extract_timestamp(f):
    name = os.path.basename(f)
    time_part = name.split("_")[-1].replace(".tdms", "")
    return pd.to_datetime(time_part, format="%Y%m%d%H%M%S")

FAULT_FREQS = [140, 93, 78]  # 고장 관련 주파수 (Hz)

def compute_selected_frequency_indices(file_list, channels, top_n=20, sampling_rate=25600):
    psd_by_channel = {ch: [] for ch in channels}
    rul_list = []

    for df, rul in file_list:
        for ch in channels:
            if ch not in df.columns:
                continue
            data = df[ch].values
            f, Pxx = welch(data, fs=sampling_rate)
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

    print(f"[✓] 초 단위 Spearman+Fault 기반 주파수 선택 완료 (총 {len(selected[channels[0]])}개)")
    return selected, f



# ▶ 에너지 엔트로피 계산 함수 (선택된 주파수 기반)
def energy_entropy_selected(data, selected_indices, sampling_rate=25600):
    f, Pxx = welch(data, fs=sampling_rate)
    selected = Pxx[selected_indices]
    selected = selected / np.sum(selected)
    selected = selected[selected > 0]
    return -np.sum(selected * np.log(selected))

# ▶ 특징 추출 함수
def extract_features_from_vibration(vib_df, sampling_rate=25600):
    features = {}
    for ch in vib_df.columns:
        data = vib_df[ch].values
        rms = np.sqrt(np.mean(data**2))
        f, Pxx = welch(data, fs=sampling_rate)

        features[f'{ch}_mean'] = np.mean(data)
        features[f'{ch}_std'] = np.std(data)
        features[f'{ch}_rms'] = rms
        features[f'{ch}_kurtosis'] = kurtosis(data)
        features[f'{ch}_skew'] = skew(data)
        features[f'{ch}_crest'] = np.max(np.abs(data)) / rms
        features[f'{ch}_band_power'] = np.sum(Pxx)

        if ch in SELECTED_FREQ_INDICES:
            features[f'{ch}_entropy'] = energy_entropy_selected(data, SELECTED_FREQ_INDICES[ch], sampling_rate)*3

    return features

# ▶ Validation 전용 특징 추출 함수
def extract_features_for_folder(folder_path, sampling_rate=25600):
    rows = []
    files = sorted(glob(os.path.join(folder_path, "*.tdms")), key=extract_timestamp)
    if not files:
        return pd.DataFrame()
    ts_all = [extract_timestamp(f) for f in files]
    end_t  = max(ts_all)
    for f, ts in zip(files, ts_all):
        vib_df = load_vibration_data(f)
        if vib_df.empty:
            continue
        rul = (end_t - ts).total_seconds()
        for sec_df in split_into_seconds(vib_df, sampling_rate):
            feats = extract_features_from_vibration(sec_df, sampling_rate)
            feats.update({'file': os.path.basename(f), 'RUL': rul})
            rows.append(feats)
    return pd.DataFrame(rows)

# ▶ 전체 데이터 처리 함수
SAMPLING_RATE = 25600  # 1초 = 25600개

# ❶ 초 단위로 슬라이싱
def split_into_seconds(df, sampling_rate=25600):
    one_sec = sampling_rate
    num_sec = df.shape[0] // one_sec
    return [df.iloc[i * one_sec : (i + 1) * one_sec] for i in range(num_sec)]
def process_all_sets(top_folder, top_n=20, sampling_rate=25600):
    global SELECTED_FREQ_INDICES, FREQ_VECTOR

    rows = []
    pairs = []  # (df, rul, file_name)

    channels = ["CH1", "CH2", "CH3", "CH4"]
    train_folders = sorted(glob(os.path.join(top_folder, "Train*")))

    for set_path in train_folders:
        files = sorted(glob(os.path.join(set_path, "*.tdms")), key=extract_timestamp)
        if not files:
            print(f"⚠️ {set_path} 폴더에 TDMS 파일 없음")
            continue

        ts_all = [extract_timestamp(f) for f in files]
        end_t = max(ts_all)

        for f, ts in zip(files[:-1], ts_all[:-1]):
            df = load_vibration_data(f)
            if df.empty:
                continue
            rul = (end_t - ts).total_seconds()
            pairs.append((df, rul, os.path.basename(f)))

    # Spearman 기반 주파수 선택
    pairs_for_selection = [(df, rul) for df, rul, _ in pairs]
    SELECTED_FREQ_INDICES, FREQ_VECTOR = compute_selected_frequency_indices(
        pairs_for_selection, channels, top_n=top_n, sampling_rate=sampling_rate
    )

    # 특징 추출 (1초 단위 슬라이싱)
    for df, rul, fname in pairs:
        for sec_df in split_into_seconds(df, sampling_rate):
            feats = extract_features_from_vibration(sec_df, sampling_rate)
            feats.update({'file': fname, 'RUL': rul})
            rows.append(feats)

    full_df = pd.DataFrame(rows)
    return full_df

WINDOW = 5
DATA_ROOT = r"c:/Users/조성찬/OneDrive - UOS/바탕 화면/배어링데이터"
MODEL_PATH = "rul_final2_all_sets.h5"

# ─────────────────────────────────────────────────────
# 기존 import, 함수(process_all_sets, split_into_seconds 등) 그대로 둡니다.
# WINDOW, DATA_ROOT, MODEL_PATH 도 동일합니다.
# ─────────────────────────────────────────────────────

# (1) Train 전체로만 process_all_sets 해서 spearman 기준 세팅
df_train = process_all_sets(DATA_ROOT)  
scaler   = MinMaxScaler().fit(df_train.drop(columns=['RUL','file']))

# (2) 모델 로드

model = load_model(MODEL_PATH)

# (생략) 기존 import, 함수, 모델 로드까지 동일

for vid in range(1, 7):
    folder_path = os.path.join(DATA_ROOT, f"Validation{vid}")
    df_val = extract_features_for_folder(folder_path)
    if df_val.empty:
        print(f"Validation{vid}: 데이터 없음")
        continue

    # 스케일링
    X_all = scaler.transform(df_val.drop(columns=['RUL','file']))
    if len(X_all) < WINDOW:
        print(f"Validation{vid}: 시퀀스 부족")
        continue

    preds = []  # 이 Validation의 예측값 리스트

    # 예시: 원본 offset 0~4
    for i in range(5):
        start = -WINDOW - i
        end   = None if i == 0 else -i
        seq   = X_all[start:end]
        pred  = model.predict(seq.reshape(1, WINDOW, -1), verbose=0)[0, 0]
        preds.append(pred)
        print(f"Validation{vid} (offset {i}): {pred:.1f}")

    # 예시: +600 offset 0~4
    offset_base = 9
    for i in range(5):
        start = -(offset_base + WINDOW + i)
        end   = - (offset_base + i)
        seq   = X_all[start:end]
        if len(seq) != WINDOW:
            print(f"Validation{vid} (offset {i}, +600): 시퀀스 길이 {len(seq)}로 SKIP")
            continue
        pred = model.predict(seq.reshape(1, WINDOW, -1), verbose=0)[0, 0] + 600
        preds.append(pred)
        print(f"Validation{vid} (offset {i}, +600): {pred:.1f}")

    offset_base = 19
    for i in range(5):
        start = -(offset_base + WINDOW + i)
        end   = - (offset_base + i)
        seq   = X_all[start:end]
        if len(seq) != WINDOW:
            print(f"Validation{vid} (offset {i}, +1200): 시퀀스 길이 {len(seq)}로 SKIP")
            continue
        pred = model.predict(seq.reshape(1, WINDOW, -1), verbose=0)[0, 0] + 1200
        preds.append(pred)
        print(f"Validation{vid} (offset {i}, +1200): {pred:.1f}")


    # preds에 10개의 예측값이 담긴 뒤

    # (1) 전체 평균
    avg_all = sum(preds) / len(preds)

    # (2) 최소·최대 제외한 평균
    sorted_preds = sorted(preds)
    trimmed_preds = sorted_preds[1:-2]            # 첫·마지막(최소·최대) 제거
    avg_trimmed = sum(trimmed_preds) / len(trimmed_preds)

    print(f"전체 {len(preds)}개 평균: {avg_all:.1f}")
    print(f"최소·최대 제외한 {len(trimmed_preds)}개 평균: {avg_trimmed:.1f}")
