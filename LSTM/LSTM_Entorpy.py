# rul_model_trainer_all_sets.py
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

Spearman 도입 x
'''
import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 에너지 엔트로피 계산 함수
def energy_entropy(data, sampling_rate=25600, n_bins=100):
    f, Pxx = welch(data, fs=sampling_rate)
    Pxx = Pxx / np.sum(Pxx)  # 정규화
    Pxx = Pxx[Pxx > 0]  # 로그 0 방지
    entropy = -np.sum(Pxx * np.log(Pxx))
    return entropy

#  진동 특징 추출 함수

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
        features[f'{ch}_entropy'] = energy_entropy(data, sampling_rate)
    return features

#  TDMS 파일에서 진동 데이터 불러오기
from nptdms import TdmsFile

def load_vibration_data(file_path):
    tdms_file = TdmsFile.read(file_path)
    group_name = tdms_file.groups()[0].name  # Vibration group
    vib_channels = tdms_file[group_name].channels()
    vib_data = {ch.name: ch.data for ch in vib_channels}
    return pd.DataFrame(vib_data)

def extract_timestamp(f):
    name = os.path.basename(f)
    time_part = name.split("_")[-1].replace(".tdms", "")  # "20160325122639"
    return pd.to_datetime(time_part, format="%Y%m%d%H%M%S")

# 메인: 특정 + RUL 생성 + 하나로 통합

def process_all_sets(top_folder):
    all_rows = []
    set_folders = sorted([os.path.join(top_folder, d) for d in os.listdir(top_folder) if os.path.isdir(os.path.join(top_folder, d))])

    for set_path in set_folders:
        print(f"\n처리 중인 폴더: {set_path}")
        tdms_files = sorted(glob(os.path.join(set_path, "*.tdms")), key=extract_timestamp)
        if not tdms_files:
            print("   .tdms 파일 없음 → 건너뜀")
            continue

        total_files = len(tdms_files)
        print(f"   총 {total_files}개 파일 발견")

        for i, file_path in enumerate(tdms_files):
            print(f"     파일 {i+1}: {os.path.basename(file_path)}")
            try:
                vib_df = load_vibration_data(file_path)
                print(f"      🔹 채널 수: {vib_df.shape[1]}개 / 샘플 수: {vib_df.shape[0]}")
                print(f"      🔹 채널 목록: {vib_df.columns.tolist()}")

                if vib_df.empty:
                    print(f"      데이터 비어있음 → 건너뜀")
                    continue

                features = extract_features_from_vibration(vib_df)
                features['file'] = os.path.basename(file_path)
                features['RUL'] = (total_files - 1 - i) * 10
                all_rows.append(features)

            except Exception as e:
                print(f"       오류 발생: {file_path}")
                print("         이유:", str(e))

    return pd.DataFrame(all_rows)

#  시퀀스 구성 함수
def create_sequences(X, y, window_size=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i + window_size - 1])
    return np.array(X_seq), np.array(y_seq)

#  프로세스 시작
if __name__ == "__main__":
    DATA_ROOT = r"c:/Users/조성찬/OneDrive - UOS/바탕 화면/배어링데이터"

    print("\n 진동 특징 추출 및 RUL 생성 중...")
    full_df = process_all_sets(DATA_ROOT)
    # full_df 검증 추가
    if full_df.empty:
        print(" full_df가 비어 있습니다. 진동 데이터에서 아무 특징도 추출되지 않았습니다.")
        exit()

    missing = [col for col in ["RUL", "file"] if col not in full_df.columns]
    if missing:
        print(f" full_df에 다음 컬럼이 없습니다: {missing}")
        print(" 현재 컬럼 목록:", full_df.columns.tolist())
        print(" 이유: 파일 처리 중 일부 tdms가 열리지 않았거나 특징 추출 실패했을 수 있습니다.")
        exit()

    print("\n 스케일링 및 시퀀스 구성 중...")
    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(full_df.drop(columns=["RUL", "file"]))
    y_all = full_df["RUL"].values
    print("\n full_df 내용 확인:")
    print(full_df.head())
    print("\n full_df 컬럼 목록:")
    print(full_df.columns.tolist())
    print("\n full_df 전체 행 수:", len(full_df))
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