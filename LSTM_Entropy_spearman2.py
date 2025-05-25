# rul_model_trainer_entropy_only.py (엔트로피만 사용하는 간소화 모델)
'''
이건 아닌거같다(엔트로피만 남기지는않기) 성능이 너무 떨어짐  다른값들과 함께 쓰는것이 좋겟음

'''
import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import spearmanr
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from nptdms import TdmsFile

SELECTED_FREQ_INDICES = {}
FREQ_VECTOR = None

# ▶ TDMS 데이터 로딩
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

# ▶ 상위 N개 주파수 선택 (Spearman)
def compute_selected_frequency_indices(file_list, channels, top_n=30, sampling_rate=25600):
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
        top_indices = np.argsort(rho_list)[-top_n:]
        selected[ch] = top_indices.tolist()

    print(f"✅ 채널별 상위 {top_n}개 주파수 선택 완료")
    return selected, f

# ▶ 에너지 엔트로피 계산
def energy_entropy_selected(data, selected_indices, sampling_rate=25600):
    f, Pxx = welch(data, fs=sampling_rate)
    selected = Pxx[selected_indices]
    selected = selected / np.sum(selected)
    selected = selected[selected > 0]
    return -np.sum(selected * np.log(selected))

# ▶ 엔트로피만 추출
def extract_entropy_features(vib_df, sampling_rate=25600):
    features = {}
    for ch in vib_df.columns:
        if ch in SELECTED_FREQ_INDICES:
            data = vib_df[ch].values
            features[f'{ch}_entropy'] = energy_entropy_selected(data, SELECTED_FREQ_INDICES[ch], sampling_rate)
    return features

# ▶ 전체 데이터 처리 (엔트로피만)
def process_all_sets_entropy_only(top_folder):
    global SELECTED_FREQ_INDICES, FREQ_VECTOR
    all_rows = []
    rul_pairs = []
    channels = ["CH1", "CH2", "CH3", "CH4"]
    set_folders = sorted([os.path.join(top_folder, d) for d in os.listdir(top_folder) if os.path.isdir(os.path.join(top_folder, d))])

    for set_path in set_folders:
        tdms_files = sorted(glob(os.path.join(set_path, "*.tdms")), key=extract_timestamp)
        total_files = len(tdms_files)
        for i, file_path in enumerate(tdms_files):
            rul = (total_files - 1 - i) * 10
            rul_pairs.append((file_path, rul))

    SELECTED_FREQ_INDICES, FREQ_VECTOR = compute_selected_frequency_indices(rul_pairs, channels, top_n=30)

    for file_path, rul in rul_pairs:
        try:
            vib_df = load_vibration_data(file_path)
            if vib_df.empty:
                continue
            features = extract_entropy_features(vib_df)
            features['file'] = os.path.basename(file_path)
            features['RUL'] = rul
            all_rows.append(features)
        except Exception as e:
            print(f" 오류 발생: {file_path} - {e}")

    return pd.DataFrame(all_rows)

# ▶ 시퀀스 구성
def create_sequences(X, y, window_size=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i + window_size - 1])
    return np.array(X_seq), np.array(y_seq)

# ▶ 메인 실행부
if __name__ == "__main__":
    DATA_ROOT = r"c:/Users/조성찬/OneDrive - UOS/바탕 화면/배어링데이터"

    print("\n 엔트로피 기반 특징 추출 중...")
    full_df = process_all_sets_entropy_only(DATA_ROOT)
    if full_df.empty:
        print(" full_df 비어 있음")
        exit()

    print("\n 스케일링 및 시퀀스 구성 중...")
    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(full_df.drop(columns=["RUL", "file"]))
    y_all = full_df["RUL"].values

    X_seq, y_seq = create_sequences(X_all, y_all, window_size=5)
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    print("\n LSTM 학습 시작...")
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

    model.save("rul_entropy_only_lstm.h5")
    print("\n 모델이 rul_entropy_only_lstm.h5 로 저장되었습니다.")
