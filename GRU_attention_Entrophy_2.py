# rul_model_trainer_all_sets.py (with log1p RUL, weighted loss, clipped output + channel-scaled GRU+Attention)

'''
▶ 최종 전략 요약
--------------------------------------------------
- Welch 스펙트럼 분석: nperseg=4096
- 채널 스케일러(ChannelScaler): 각 채널별 γ 가중치 적용
    · γ 초기값 5.0, 학습률은 5배
    · 전체 채널 feature에 곱해지는 방식 (mean, std, entropy 등 모두 영향)
- entropy feature 강조: 각 채널의 entropy에 대해 10제곱 가중치 적용
- CustomModel: γ 파라미터에만 별도 gradient 스케일링 적용
- log1p(RUL), weighted MAE, 클리핑(min=10) 등 RUL 안정성 향상 전략 적용
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Input, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from nptdms import TdmsFile
import matplotlib.pyplot as plt

SELECTED_FREQ_INDICES = {}
FREQ_VECTOR = None
FAULT_FREQS = [140, 93, 78, 6.7]

# ──────────────────── ❶ 채널 스케일러 레이어 정의 ────────────────────


class ChannelScaler(Layer):
    def __init__(self, n_channels=4, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.gamma = self.add_weight(
            shape=(1, 1, n_channels, 1),
            initializer=tf.keras.initializers.Constant(1.0),  # ✅ 초기값을 3으로 변경
            trainable=True,
            name="gamma_channel"
        )

    def call(self, x):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        F = x.shape[2]
        f_per_ch = F // self.n_channels
        x4d = tf.reshape(x, (B, T, self.n_channels, f_per_ch))
        x4d = x4d * self.gamma
        return tf.reshape(x4d, (B, T, F))

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_channels": self.n_channels
        })
        return config

# ──────────────────── ❸ CustomModel 정의: gamma에 더 큰 학습률 적용 ────────────────────
class CustomModel(tf.keras.Model):
    def __init__(self, *args, scaler_layer_name="ch_scaler", gamma_lr_scale=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler_layer_name = scaler_layer_name
        self.gamma_lr_scale = gamma_lr_scale

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        # 모든 trainable 변수에 대해 gradient 계산
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)

        # gamma만 스케일 조정
        scaled_grads = []
        for var, grad in zip(trainable_vars, grads):
            if self.scaler_layer_name in var.name and "gamma_channel" in var.name:
                grad = grad * self.gamma_lr_scale  # 🔥 gamma 학습률 스케일링
            scaled_grads.append(grad)

        self.optimizer.apply_gradients(zip(scaled_grads, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# ──────────────────── ❷ 모델 정의 (GRU + Attention + Channel Scale) ────────────────────
def build_gru_attention_model(seq_len, n_feat, gru_units=64, n_heads=4, attn_key_dim=32, ff_dim=32, dropout_rate=0.3, n_channels=4):
    inp = Input(shape=(seq_len, n_feat))
    x = ChannelScaler(n_channels=n_channels, name="ch_scaler")(inp)
    x = GRU(gru_units, return_sequences=True)(x)
    att = MultiHeadAttention(num_heads=n_heads, key_dim=attn_key_dim, dropout=0.0)(x, x, x)  # Dropout 제거
    x = LayerNormalization(epsilon=1e-6)(x + att)
    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(gru_units)(ffn)
    x = LayerNormalization(epsilon=1e-6)(x + ffn)
    # Dropout 제거됨
    x = GlobalAveragePooling1D()(x)
    out = Dense(1)(x)
    return Model(inp, out, name="GRU_Attn_ChScale")

# ──────────────────── 손실 함수 ────────────────────
def weighted_mae(y_true, y_pred):
    weights = 1 / (K.clip(y_true, 1, np.inf))
    return K.mean(weights * K.abs(y_true - y_pred))

# ──────────────────── 유틸리티 함수 ────────────────────
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
        ENTROPY_WEIGHTS = {"CH1": 3, "CH2": 3, "CH3": 3, "CH4": 3}
        if ch in SELECTED_FREQ_INDICES:
            features[f'{ch}_entropy'] = energy_entropy_selected(data, SELECTED_FREQ_INDICES[ch]) * ENTROPY_WEIGHTS.get(ch, 1)
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
        for file_path, ts in zip(tdms_files[:-1], file_times[:-1]):
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


# 추가: 평가 지표 함수 정의

def compute_percent_error(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    nonzero_mask = actual != 0
    eri = np.zeros_like(actual)
    eri[nonzero_mask] = 100 * (actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask]
    return eri

def compute_arul_score(eri):
    eri = np.array(eri)
    score = np.where(
        eri <= 0,
        np.exp(-np.log(0.5) * eri / 20),
        np.exp(+np.log(0.5) * eri / 50)
    )
    return score

if __name__ == "__main__":
    DATA_ROOT = r"c:/Users/조성찬/OneDrive - UOS/바탕 화면/배어링데이터"
    full_df = process_all_sets(DATA_ROOT)
    full_df = full_df.sort_values(by='file')

    if full_df.empty:
        exit()


    # 🔥 ❶ RUL이 0인 샘플 제거
    full_df = full_df[full_df["RUL"] > 0].reset_index(drop=True)

    # 🔥 ❷ Hold-out 샘플 분리 (고장 직전 파일)
    holdout_rows = full_df.groupby(full_df['file'].str.extract(r'(Train\d+)')[0]) \
                            .apply(lambda g: g.loc[g['RUL'].idxmin()]) \
                            .reset_index(drop=True)
    train_val_df = pd.concat([full_df, holdout_rows]).drop_duplicates(keep=False)

    # 🔥 ❸ Stratified Split을 위한 RUL 구간 설정
    bins = [-1, 30, 300, 2000, np.inf]
    labels = pd.cut(train_val_df["RUL"], bins=bins, labels=False)

    # 🔥 ❹ 시퀀스 생성
    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(train_val_df.drop(columns=["RUL", "file"]))
    y_all_log = np.log1p(train_val_df["RUL"].values)
    X_seq, y_seq = create_sequences(X_all, y_all_log, window_size=5)
    labels_seq = labels[4:]  # 시퀀스 앞부분 제거분 만큼 제외

    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=labels_seq
    )

        # 🔥 ❸ 모델 학습
    base_model = build_gru_attention_model(
        seq_len=X_train.shape[1],
        n_feat=X_train.shape[2],
        n_channels=4
    )

    model = CustomModel(inputs=base_model.input, outputs=base_model.output)
    model.compile(optimizer='adam', loss=weighted_mae)
    model.fit(X_train, y_train, epochs=5000, batch_size=16, validation_data=(X_val, y_val))

    # ❹ 평가 
    pred_log = model.predict(X_val)
    y_val_true = np.expm1(y_val)
    pred = np.expm1(pred_log)

    valid_mask = y_val_true > 0
    filtered_y_true = y_val_true[valid_mask]
    filtered_pred = pred.flatten()[valid_mask]

    eri = compute_percent_error(filtered_y_true, filtered_pred)
    a_rul_scores = compute_arul_score(eri)

    print(f"\n [전체 기준] 평균 상대 오차 (MARE): {np.mean(np.abs(eri)):.2f}%")
    print(f"정확도 점수 (A_RUL 평균): {np.mean(a_rul_scores):.4f}")


    # 🔥 ❺ 저장
    model.save("rul_lstm_all_sets.h5")
    print("\n모델이 rul_lstm_all_sets.h5로 저장되었습니다.")
    gamma = model.get_layer("ch_scaler").gamma.numpy().flatten()
    print("학습된 채널 가중치 γ:", gamma)
