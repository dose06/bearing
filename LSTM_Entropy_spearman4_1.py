# rul_model_trainer_all_sets.py – LSTM + ChannelEntropyScaler (채널별 엔트로피 가중치 학습)
"""
■ 목적
    • CH1 ~ CH4 엔트로피(feature `CHx_entropy`)에 학습 가능한 γ₁‥γ₄를 곱해 채널 중요도를 모델이 스스로 학습하도록 한다.
    • 시퀀스 모델은 순수 LSTM만 사용한다.

■ 파이프라인
    1) Spearman + 고장 주파수 기반 주파수 선택 → 특징 추출
    2) DataFrame → MinMaxScaler → 시퀀스 생성 (window=5)
    3) Custom Layer `ChannelEntropyScaler` 로 엔트로피 열 4개에 γ 곱하기
    4) LSTM(64) → Dense(32) → 출력
    5) 학습 완료 후 γ 값 출력 및 모델 저장
"""

#────────────────────────────────────────────
# 0. 공용 import & 상수
#────────────────────────────────────────────
import os, numpy as np, pandas as pd, tensorflow as tf
from glob import glob
from scipy.signal import welch
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from nptdms import TdmsFile

CHANNELS      = ["CH1", "CH2", "CH3", "CH4"]
FAULT_FREQS   = [140, 93, 78]
ENTROPY_COLS  = [f"{ch}_entropy" for ch in CHANNELS]
SAMPLING_RATE = 25600
WINDOW_SIZE   = 5

#────────────────────────────────────────────
# 1. 데이터 로드 & 전처리 함수
#────────────────────────────────────────────

def load_vibration_data(file_path: str) -> pd.DataFrame:
    tdms = TdmsFile.read(file_path)
    grp  = tdms.groups()[0].name
    return pd.DataFrame({c.name: c.data for c in tdms[grp].channels()})

def extract_timestamp(fp: str) -> pd.Timestamp:
    return pd.to_datetime(os.path.basename(fp).split("_")[-1].replace(".tdms", ""), format="%Y%m%d%H%M%S")

def compute_selected_frequency_indices(file_pairs, top_n=10):
    psd_by_ch, rul_list = {c: [] for c in CHANNELS}, []
    for fp, rul in file_pairs:
        df = load_vibration_data(fp)
        if df.empty: continue
        for ch in CHANNELS:
            if ch not in df.columns: continue
            f, Pxx = welch(df[ch].values, fs=SAMPLING_RATE)
            psd_by_ch[ch].append(Pxx)
        rul_list.append(rul)
    selected = {}
    for ch in CHANNELS:
        mat = np.array(psd_by_ch[ch])
        if mat.size == 0: continue
        rho = [abs(spearmanr(mat[:, i], rul_list)[0]) for i in range(mat.shape[1])]
        top_idx   = np.argsort(rho)[-top_n:]
        fault_idx = [np.argmin(np.abs(f - ff)) for ff in FAULT_FREQS]
        selected[ch] = sorted(set(top_idx.tolist() + fault_idx))
    return selected

def energy_entropy_selected(sig, sel_idx):
    f, Pxx = welch(sig, fs=SAMPLING_RATE)
    sel    = Pxx[sel_idx]
    sel    = sel / np.sum(sel)
    sel    = sel[sel > 0]
    return -np.sum(sel * np.log(sel))

def extract_features_from_vibration(df: pd.DataFrame, sel_dict: dict) -> dict:
    feat = {}
    for ch in df.columns:
        x = df[ch].values
        f, Pxx = welch(x, fs=SAMPLING_RATE)
        rms = np.sqrt(np.mean(x**2))
        feat.update({
            f"{ch}_mean"      : np.mean(x),
            f"{ch}_std"       : np.std(x),
            f"{ch}_rms"       : rms,
            f"{ch}_kurtosis"  : kurtosis(x),
            f"{ch}_skew"      : skew(x),
            f"{ch}_crest"     : np.max(np.abs(x)) / rms,
            f"{ch}_band_power": np.sum(Pxx)
        })
        if ch in sel_dict:
            feat[f"{ch}_entropy"] = energy_entropy_selected(x, sel_dict[ch])
    return feat

def process_all_sets(root: str) -> pd.DataFrame:
    rows, pairs = [], []
    for set_path in sorted([os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]):
        files = sorted(glob(os.path.join(set_path, "*.tdms")), key=extract_timestamp)
        if not files: continue
        times = [extract_timestamp(f) for f in files]
        end_t = max(times)
        pairs.extend([(f, (end_t - t).total_seconds()) for f, t in zip(files, times)])
    sel_dict = compute_selected_frequency_indices(pairs)
    for fp, rul in pairs:
        try:
            df = load_vibration_data(fp);  rows.append({**extract_features_from_vibration(df, sel_dict), "file": os.path.basename(fp), "RUL": rul})
        except Exception as e:
            print("⚠", fp, e)
    return pd.DataFrame(rows)

def create_sequences(X, y, win=5):
    Xs, ys = [], []
    for i in range(len(X) - win + 1):
        Xs.append(X[i:i+win]);  ys.append(y[i+win-1])
    return np.array(Xs), np.array(ys)

#────────────────────────────────────────────
# 2. Custom Layer & LSTM 모델
#────────────────────────────────────────────
class ChannelEntropyScaler(tf.keras.layers.Layer):
    def __init__(self, entropy_indices, **kw):
        super().__init__(**kw)
        self.idx   = entropy_indices
        self.gamma = self.add_weight(
            shape=(4,), initializer=tf.keras.initializers.Constant(2.0),
            trainable=True, name="gamma_entropy"
        )

    def call(self, x):
        xs = tf.unstack(x, axis=-1)
        for i, col in enumerate(self.idx):
            xs[col] = xs[col] * self.gamma[i]
        return tf.stack(xs, axis=-1)

    # ★ 추가: 직렬화를 위한 get_config ★
    def get_config(self):
        config = super().get_config()
        config.update({"entropy_indices": self.idx})
        return config

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_model(seq_len, n_feat, entropy_indices):
    inp = Input(shape=(seq_len, n_feat))
    x   = ChannelEntropyScaler(entropy_indices, name="entropy_scaler")(inp)
    x   = LSTM(64)(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    return Model(inp, out)

#────────────────────────────────────────────
# 3. 메인 실행부
#────────────────────────────────────────────
if __name__ == "__main__":
    DATA_ROOT = r"c:/Users/조성찬/OneDrive - UOS/바탕 화면/배어링데이터"

    print("\n▣ 특징 추출 중…")
    df = process_all_sets(DATA_ROOT).sort_values("file")
    if df.empty: raise SystemExit("❗ DataFrame empty")

    feat_cols   = [c for c in df.columns if c not in ("RUL", "file")]
    entropy_idx = [feat_cols.index(c) for c in ENTROPY_COLS]

    X_scaled = MinMaxScaler().fit_transform(df[feat_cols])
    y        = df["RUL"].values
    X_seq, y_seq = create_sequences(X_scaled, y, WINDOW_SIZE)
    X_tr, X_val, y_tr, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    print("▣ 모델 학습 시작…")
    model = build_model(X_tr.shape[1], X_tr.shape[2], entropy_idx)
    model.compile(optimizer=Adam(5e-4), loss="mae")
    model.fit(X_tr, y_tr, epochs=5000, batch_size=16, validation_data=(X_val, y_val))

    print("▣ 평가…")
    pred = model.predict(X_val).flatten()
    mask = y_val != 0
    mare = np.mean(np.abs((y_val[mask] - pred[mask]) / y_val[mask])) * 100
    print(f"MARE: {mare:.2f}%")

    print("▣ 학습된 γ 값:", model.get_layer("entropy_scaler").gamma.numpy())
    model.save("rul_lstm_all_sets.h5")
    print("\n모델이 rul_lstm_all_sets.h5로 저장되었습니다.")
