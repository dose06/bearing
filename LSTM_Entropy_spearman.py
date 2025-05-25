# rul_model_trainer_all_sets.py (with Spearman frequency selection)
'''
진동 데이터 (시간 흐름)

↓
FFT → 주파수별 진폭

**(진동 신호는 시간 도메인에서의 파형인데, 이걸 주파수 도메인으로 바꾸면, 어떤 주파수에 에너지가 얼마나 분포돼 있는지를 알 수 있음)**

↓
정규화된 주파수 에너지

↓
선택된 주파수 (Spearman 상관↓)

↓
정보 엔트로피 계산 (에너지 엔트로피)

↓
단조 감소하는 특징 신호

↓
→ 수명 예측 모델의 입력(RUL Regression Target)

Spearman 도입하였지만 제대로 도입 x

'''
import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import kurtosis, skew, spearmanr
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from nptdms import TdmsFile

# ▶ Spearman 기반 선택된 주파수 인덱스 초기화
SELECTED_FREQ_INDICES = None
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

# ▶ Spearman 기반 주파수 선택 함수
def compute_selected_frequency_indices(file_list, channel="CH1", threshold=0.3, sampling_rate=25600):
    psd_matrix, rul_list = [], []
    for file_path, rul in file_list:
        df = load_vibration_data(file_path)
        if df.empty or channel not in df.columns:
            continue
        data = df[channel].values
        f, Pxx = welch(data, fs=sampling_rate)
        psd_matrix.append(Pxx)
        rul_list.append(rul)

    psd_matrix = np.array(psd_matrix)
    rul_array = np.array(rul_list)
    selected = [i for i in range(psd_matrix.shape[1]) if abs(spearmanr(psd_matrix[:, i], rul_array)[0]) >= threshold]
    print(f"✅ Spearman 상관 기반 선택된 주파수 개수: {len(selected)}")
    return selected, f

# ▶ 에너지 엔트로피 계산 함수 (선택된 주파수 기반)
def energy_entropy_selected(data, sampling_rate=25600):
    global SELECTED_FREQ_INDICES
    f, Pxx = welch(data, fs=sampling_rate)
    selected = Pxx[SELECTED_FREQ_INDICES]
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

        if SELECTED_FREQ_INDICES and ch == "CH1":
            features[f'{ch}_entropy'] = energy_entropy_selected(data, sampling_rate)

    return features

# ▶ 전체 데이터 처리 함수
def process_all_sets(top_folder):
    global SELECTED_FREQ_INDICES, FREQ_VECTOR
    all_rows = []
    rul_pairs = []
    set_folders = sorted([os.path.join(top_folder, d) for d in os.listdir(top_folder) if os.path.isdir(os.path.join(top_folder, d))])

    for set_path in set_folders:
        tdms_files = sorted(glob(os.path.join(set_path, "*.tdms")), key=extract_timestamp)
        total_files = len(tdms_files)

        for i, file_path in enumerate(tdms_files):
            rul = (total_files - 1 - i) * 10
            rul_pairs.append((file_path, rul))

    # Spearman 상관 기반 선택 주파수 계산 (CH1 기준)
    SELECTED_FREQ_INDICES, FREQ_VECTOR = compute_selected_frequency_indices(rul_pairs)

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
            print(f"❌ 오류 발생: {file_path} - {e}")

    return pd.DataFrame(all_rows)

# ▶ 시퀀스 구성 함수
def create_sequences(X, y, window_size=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i + window_size - 1])
    return np.array(X_seq), np.array(y_seq)

# ▶ 메인 실행부
if __name__ == "__main__":
    DATA_ROOT = r"c:/Users/조성찬/OneDrive - UOS/바탕 화면/배어링데이터"

    print("\n 진동 특징 추출 및 RUL 생성 중...")
    full_df = process_all_sets(DATA_ROOT)

    if full_df.empty:
        print("❌ full_df가 비어 있습니다.")
        exit()

    print("\n 스케일링 및 시퀀스 구성 중...")
    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(full_df.drop(columns=["RUL", "file"]))
    y_all = full_df["RUL"].values

    X_seq, y_seq = create_sequences(X_all, y_all, window_size=5)
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    print("\n LSTM 모델 학습 시작...")
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val))

    pred = model.predict(X_val)
    print("\n 평가 결과:")
    print("MAE:", mean_absolute_error(y_val, pred))
    print("R²:", r2_score(y_val, pred))

    model.save("rul_lstm_all_sets.h5")
    print("\n모델이 rul_lstm_all_sets.h5로 저장되었습니다.")