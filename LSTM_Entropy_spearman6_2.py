# rul_model_trainer_all_sets.py (Spearman 개선: 낮은 threshold + 상위 N개 선택 + 다채널)
"""
📁 rul_model_trainer_all_sets.py

🎯 목적:
    - 다채널 진동 데이터를 활용하여 베어링의 남은 수명(RUL, Remaining Useful Life)을 예측하는 딥러닝 모델(LSTM 기반)을 학습하고 평가함.

📌 주요 기능:
    1. Welch 스펙트럼 분석 → Spearman 상관계수 기반 중요 주파수 선택
    2. 고장 관련 주파수 포함
    3. 에너지 엔트로피 기반 특징 추출 (강조: entropy * 3)
    4. 각 채널(CH1~CH4)의 통계/스펙트럼 특징을 벡터로 병렬 결합
    5. 1초 단위로 슬라이싱한 데이터에 대해 시퀀스 구성
    6. log1p(RUL) 변환 적용하여 안정적 학습
    7. 과소예측에 높은 페널티를 주는 A_RUL 근사 손실 함수 사용 (`approx_arul_loss`)
    8. Stratified split 적용 (RUL 구간 기준)
    9. 학습된 모델의 예측 성능 평가 (MARE, A_RUL)

🔧 구성 요소:
    - Feature 추출: mean, std, rms, kurtosis, skew, crest, band_power, entropy
    - 모델 구조: LSTM → Dense → Dense(1)
    - 손실 함수: approx_arul_loss (A_RUL 점수 최적화 목적)

💾 출력:
    - rul_lstm_all_sets.h5: 학습된 모델 저장
    - 평균 상대 오차 (MARE) 및 A_RUL 점수 출력

🧪 참고:
    - log1p, expm1 변환으로 안정적 예측 및 역변환 수행
    - hold-out: 각 Train셋의 마지막 TDMS 샘플은 학습에 포함되지 않음
"""
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
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import backend as K


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
            f, Pxx = welch(data, fs=sampling_rate, nperseg=4096)
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
    f, Pxx = welch(data, fs=sampling_rate, nperseg=4096)
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
        f, Pxx = welch(data, fs=sampling_rate, nperseg=4096)

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
SAMPLING_RATE = 25600  # 1초 = 25600개

# ❶ 초 단위로 슬라이싱
def split_into_seconds(df, sampling_rate=25600):
    one_sec = sampling_rate
    num_sec = df.shape[0] // one_sec
    return [df.iloc[i * one_sec : (i + 1) * one_sec] for i in range(num_sec)]

# ---------- 전체 데이터 로딩 (마지막 TDMS는 hold-out) ---------- ★변경
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

# ✔️ 가중치 손실 함수 정의 (log1p 적용한 상태에서 사용)
def weighted_mae(y_true, y_pred):
    weights = 1 / (K.clip(y_true, 1, np.inf))  # log1p 된 값 기준
    return K.mean(weights * K.abs(y_true - y_pred))


def approx_arul_loss(y_true, y_pred):
    error = y_true - y_pred
    loss = tf.where(
        error <= 0,
        tf.exp(-K.log(0.5) * error / 20.0),
        tf.exp(K.log(0.5) * error / 50.0)
    )
    return 10000000000 * tf.reduce_mean(loss)  # 👉 스케일을 키워서 gradient 증폭



# ▶ 시퀀스 구성 함수
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

WINDOW = 5                               # 시퀀스 길이


if __name__ == "__main__":
    DATA_ROOT = r"c:/Users/조성찬/OneDrive - UOS/바탕 화면/배어링데이터"
    print("\n📦 진동 특징 추출 및 RUL 생성 중...")
    
    full_df = process_all_sets(DATA_ROOT)  # holdout_list 안 씀
    full_df = full_df.sort_values(by='file')

    if full_df.empty:
        print("❌ full_df가 비어 있습니다.")
        exit()

    # ❶ 고장 직전(최소 RUL) 파일 → hold-out
    holdout_rows = full_df.groupby(full_df['file'].str.extract(r'(Train\d+)')[0]) \
                          .apply(lambda g: g.loc[g['RUL'].idxmin()]) \
                          .reset_index(drop=True)

    train_val_df = pd.concat([full_df, holdout_rows]).drop_duplicates(keep=False)

    print("\n🧪 스케일링 및 시퀀스 구성 중...")
    scaler = MinMaxScaler()
    
    # ❷ stratify 라벨 생성
    bins   = [-1, 30, 300, 2000, np.inf]
    labels = pd.cut(train_val_df["RUL"], bins=bins, labels=False)

    # 시퀀스 구성
    X_all = scaler.fit_transform(train_val_df.drop(columns=["RUL", "file"]))
    # ✔️ log1p(RUL) 적용
    y_all_log = np.log1p(train_val_df["RUL"].values)

    # 시퀀스 구성
    X_seq, y_seq = create_sequences(X_all, y_all_log, window_size=5)

    labels_seq   = labels[5-1:].to_numpy()  # WINDOW = 5

    # stratified split
    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        X_seq, y_seq, labels_seq,
        test_size=0.1, random_state=42, stratify=labels_seq
    )
    print(f"\n📏 전체 row 수: {len(full_df)}개, 시퀀스 수: {len(X_seq)}개")
    print("\n🧠 LSTM 모델 학습 시작...")
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss= approx_arul_loss)
    model.fit(X_train, y_train, epochs=1000, batch_size=16, validation_data=(X_val, y_val))

    
    print("\n📊 검증 데이터 예측 중...")
    pred_log = model.predict(X_val)
    pred_log = np.clip(pred_log, 0, 15)  # 15 넘어가면 자름
    y_val_true = np.expm1(y_val)
    pred = np.expm1(pred_log).flatten()
    actual = y_val_true

    # 오차 및 평가 점수
    eri = compute_percent_error(actual, pred)
    a_rul = compute_arul_score(eri)

    print(f"\n✅ 평균 상대 오차 (MARE): {np.mean(np.abs(eri)):.2f}%")
    print(f"✅ A_RUL 평균 점수     : {np.mean(a_rul):.4f}")

    model.save("rul_lstm_all_sets.h5")
    print("\n💾 모델이 'rul_lstm_all_sets.h5'로 저장되었습니다.")
    
    import matplotlib.pyplot as plt

    plt.scatter(actual, pred, alpha=0.6)
    plt.plot([0, max(actual)], [0, max(actual)], 'r--')  # y=x 기준선
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("RUL 예측 분포")
    plt.grid(True)
    plt.show()