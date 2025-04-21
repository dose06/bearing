# rul_model_trainer_all_sets.py (Spearman 개선: 낮은 threshold + 상위 N개 선택 + 다채널)
'''
채널 4개(CH1 ~ CH4) → 어떻게 처리하고 결합했는가?

### 결합 방식

각 채널에서 추출한 특징들을 **하나의 벡터**로 **병렬 결합(concatenate)**합니다.

CH1_mean, CH1_std, CH1_entropy, ..., CH4_band_power, CH4_entropy

—>>>>>>그렇다면 엔트로피의 비중이 낮으므로 키우자( 엔트로피값 *3)
->>>>>> ch별 상위 10개 추출
->>>>>> 초단위 라벨링
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

# ▶ Spearman 기반 주파수 선택 함수 (상위 N개 고정)
def compute_selected_frequency_indices(file_list, channels, top_n=10, sampling_rate=25600):
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
            f, Pxx = welch(data, fs=sampling_rate)
            psd_by_channel[ch].append(Pxx)
        rul_list.append(rul)

    selected = {}
    for ch in channels:
        psd_matrix = np.array(psd_by_channel[ch])
        if psd_matrix.shape[0] == 0:
            continue
        rho_list = [abs(spearmanr(psd_matrix[:, i], rul_list)[0]) for i in range(psd_matrix.shape[1])]
        top_indices = np.argsort(rho_list)[-top_n:]  # 상위 N개 선택
        selected[ch] = top_indices.tolist()

    print(f" Spearman 기반 주파수 선택 완료 (채널별 {top_n}개)")
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

# ▶ 전체 데이터 처리 함수
def process_all_sets(top_folder):
    global SELECTED_FREQ_INDICES, FREQ_VECTOR
    all_rows = []
    rul_pairs = []
    channels = ["CH1", "CH2", "CH3", "CH4"]
    set_folders = sorted([os.path.join(top_folder, d) for d in os.listdir(top_folder) if os.path.isdir(os.path.join(top_folder, d))])

    for set_path in set_folders:
        tdms_files = sorted(glob(os.path.join(set_path, "*.tdms")), key=extract_timestamp)
        if not tdms_files:
            print(f"⚠️ {set_path} 폴더에 .tdms 파일이 없습니다. 건너뜁니다.")
            continue  # 이 폴더는 건너뜀
        
        file_times = [extract_timestamp(f) for f in tdms_files]
        end_time = max(file_times)

        for file_path, ts in zip(tdms_files, file_times):
            rul = (end_time - ts).total_seconds()  # 🔥 초 단위 RUL
            rul_pairs.append((file_path, rul))


    SELECTED_FREQ_INDICES, FREQ_VECTOR = compute_selected_frequency_indices(rul_pairs, channels, top_n=10)

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
        print(" full_df가 비어 있습니다.")
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
    model.fit(X_train, y_train, epochs=5000, batch_size=16, validation_data=(X_val, y_val))

    pred = model.predict(X_val)
    print("\n 평가 결과:")
    print("MAE:", mean_absolute_error(y_val, pred))
    print("R²:", r2_score(y_val, pred))

    model.save("rul_lstm_all_sets.h5")
    print("\n모델이 rul_lstm_all_sets.h5로 저장되었습니다.")