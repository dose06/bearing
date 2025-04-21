# -*- coding: utf-8 -*-
"""
RUL 예측용 LSTM 파이프라인  (CH2·CH3·CH4 / Spearman 주파수 선택 / 슬라이딩‑윈도우)
작성 : 2025‑04‑20
"""

# ─────────────────────────── 0. 설정 파라미터 ──────────────────────────
USE_LOG_RUL   = True          # True → log1p(RUL) , False → RUL / 1e4
WINDOW_SIZE   = 20            # 시퀀스 길이(프레임 수)
STRIDE_TRAIN  = 1             # 학습 윈도우 stride
STRIDE_VAL    = WINDOW_SIZE   # 검증 윈도우 stride
TOP_N_FREQ    = 20            # Spearman 상위 주파수 개수
ENTROPY_WT    = {"CH2": 4, "CH3": 4, "CH4": 4}
FAULT_FREQS   = [140, 93, 78, 6.7]           # (Hz)

# ─────────────────────────── 1. 라이브러리 ────────────────────────────
import os, random, warnings
import numpy as np
import pandas as pd
from glob   import glob
from scipy.signal import welch
from scipy.stats  import kurtosis, skew, spearmanr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers  import Adam
from tensorflow.keras.callbacks   import EarlyStopping, ReduceLROnPlateau
from nptdms import TdmsFile
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────── 2. 유틸리티 ──────────────────────────────
def load_vibration_data(path):
    tdms_file = TdmsFile.read(path)
    group     = tdms_file.groups()[0]
    return pd.DataFrame({ch.name: ch.data for ch in group.channels()})


def extract_timestamp(path):
    fname = os.path.basename(path)
    tstr  = fname.split("_")[-1].replace(".tdms", "")
    return pd.to_datetime(tstr, format="%Y%m%d%H%M%S")

# ─────────────────── 3. Spearman + Fault 주파수 선택 ──────────────────
SELECTED_FREQ_INDICES = {}     # 채널별 선택된 인덱스
FREQ_VECTOR           = None   # welch() 결과 f 벡터 저장


def compute_selected_indices(file_list, channels, top_n=TOP_N_FREQ, fs=25600):
    """
    file_list : [(tdms_path, RUL_sec), ...]
    반환값    : {ch : [idx1, idx2, ...]}
    """
    psd_by_ch, rul_list = {c: [] for c in channels}, []
    for path, rul in file_list:
        df = load_vibration_data(path)
        if df.empty:
            continue
        for c in channels:
            if c not in df.columns:
                continue
            _, Pxx = welch(df[c].values, fs=fs, nperseg=4096)
            psd_by_ch[c].append(Pxx)
        rul_list.append(rul)

    global FREQ_VECTOR
    FREQ_VECTOR = welch(np.zeros(4096), fs=fs, nperseg=4096)[0]

    selected = {}
    for ch in channels:
        psd_m = np.asarray(psd_by_ch[ch])
        if psd_m.size == 0:
            continue

        rho = [abs(spearmanr(psd_m[:, i], rul_list)[0])
               for i in range(psd_m.shape[1])]

        s_idx     = np.argsort(rho)[-top_n:]
        fault_idx = [np.argmin(abs(FREQ_VECTOR - ff)) for ff in FAULT_FREQS]
        total_idx = sorted(set(s_idx.tolist() + fault_idx))
        if not total_idx:
            continue

        selected[ch] = total_idx
        print(f"{ch} 선택 주파수(Hz) → {[f'{FREQ_VECTOR[i]:.1f}' for i in total_idx]}")
    return selected

# ───────────────────── 4. 특징 추출 함수 ──────────────────────────────

def band_entropy(sig, idx, fs=25600):
    _, Pxx = welch(sig, fs=fs, nperseg=4096)
    band   = Pxx[idx]
    band   = band / band.sum()
    band   = band[band > 0]
    return -np.sum(band * np.log(band))


def extract_features(df, fs=25600):
    out = {}
    for ch in df.columns:
        sig = df[ch].values
        rms = np.sqrt(np.mean(sig ** 2))
        _, Pxx = welch(sig, fs=fs, nperseg=4096)

        out[f"{ch}_mean"]  = sig.mean()
        out[f"{ch}_std"]   = sig.std()
        out[f"{ch}_rms"]   = rms
        out[f"{ch}_kurt"]  = kurtosis(sig)
        out[f"{ch}_skew"]  = skew(sig)
        out[f"{ch}_crest"] = abs(sig).max() / rms
        out[f"{ch}_band"]  = Pxx.sum()

        if ch in SELECTED_FREQ_INDICES:
            ee = band_entropy(sig, SELECTED_FREQ_INDICES[ch], fs)
            out[f"{ch}_entropy"] = ee * ENTROPY_WT[ch]
    return out

# ─────────────── 5. 모든 .tdms → 특징 DataFrame ──────────────────────

def process_sets(root, mode="train"):
    sets     = range(1, 7) if mode == "train" else range(7, 9)
    folders  = [os.path.join(root, f"Train{i}") for i in sets]
    channels = ["CH2", "CH3", "CH4"]
    if mode == "val":
        random.seed(42)

    pair, rows = [], []

    # 5‑1) 파일 목록 + RUL 파생
    for fold in folders:
        files = sorted(glob(os.path.join(fold, "*.tdms")), key=extract_timestamp)
        if not files:
            print(f"⚠️  {fold} 에 .tdms 없음")
            continue
        if mode == "val" and len(files) > 2:
            cut = random.randint(len(files) // 3, len(files) - 2)
            files = files[:cut]

        times = [extract_timestamp(f) for f in files]
        tend  = max(times)
        pair.extend([(f, (tend - t).total_seconds()) for f, t in zip(files, times)])

    # 5‑2) Spearman 주파수 선택 (1회)
    global SELECTED_FREQ_INDICES
    SELECTED_FREQ_INDICES = compute_selected_indices(pair, channels)

    # 5‑3) 특징 추출
    for f, rul in pair:
        try:
            df = load_vibration_data(f)
            if df.empty:
                raise ValueError("empty tdms")
            feats = extract_features(df)
            feats.update({"file": os.path.basename(f), "RUL": rul})
            rows.append(feats)
        except Exception as e:
            print("✖", f, e)

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        raise RuntimeError(f"{mode} 데이터 생성 실패 : 채널명 또는 파일 확인")
    return df_out

# ─────────────────── 6. 슬라이딩 윈도우 시퀀스 ────────────────────────

def sliding_windows(X, y, win=WINDOW_SIZE, stride=1):
    Xseq, yseq = [], []
    for s in range(0, len(X) - win + 1, stride):
        Xseq.append(X[s : s + win])
        yseq.append(y[s + win - 1])
    return np.asarray(Xseq, np.float32), np.asarray(yseq, np.float32)

# ────────────────────────── 7.   Main  ────────────────────────────────
if __name__ == "__main__":
    DATA_ROOT = r"c:/Users/조성찬/OneDrive - UOS/바탕 화면/배어링데이터"

    print("[1] 특징 추출 중 …")
    train_df = process_sets(DATA_ROOT, "train").sort_values("file")
    val_df   = process_sets(DATA_ROOT, "val"  ).sort_values("file")
    print("train rows:", len(train_df), "  val rows:", len(val_df))

    # 7‑1) 스케일링
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(train_df.drop(["RUL", "file"], axis=1)).astype(np.float32)
    Xva = scaler.transform( val_df.drop(["RUL", "file"], axis=1)).astype(np.float32)

    # 7‑2) RUL 변환
    ytr_raw, yva_raw = train_df["RUL"].values, val_df["RUL"].values
    if USE_LOG_RUL:
        ytr, yva = np.log1p(ytr_raw), np.log1p(yva_raw)
    else:
        ytr, yva = ytr_raw / 1e4, yva_raw / 1e4

    # 7‑3) 슬라이딩 윈도우
    Xtr_seq, ytr_seq = sliding_windows(Xtr, ytr, WINDOW_SIZE, STRIDE_TRAIN)
    Xva_seq, yva_seq = sliding_windows(Xva, yva, WINDOW_SIZE, STRIDE_VAL)
    print("train seq:", Xtr_seq.shape, "  val seq:", Xva_seq.shape)

    # ──────────────── LSTM 모델 ─────────────────
    model = Sequential([
        LSTM(32, input_shape=(WINDOW_SIZE, Xtr_seq.shape[2]), kernel_regularizer=l2(1e-4)),
        Dropout(0.30),
        Dense(16, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.20),
        Dense(1)
    ])
    model.compile(optimizer=Adam(5e-4), loss="mae")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=50, min_delta=1e-4,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8,
                          min_lr=1e-5, verbose=1)
    ]

    print("[2] 학습 시작 …")
    model.fit(Xtr_seq, ytr_seq,
              epochs=5000,
              batch_size=8,
              validation_data=(Xva_seq, yva_seq),
              callbacks=callbacks,
              verbose=2)

    # ──────────────── 평가 ─────────────────────
    pred = model.predict(Xva_seq, verbose=0).flatten()
    if USE_LOG_RUL:
        pred = np.expm1(pred)
        y_true = np.expm1(yva_seq)
    else:
        pred = pred * 1e4
        y_true = yva_seq * 1e4

    mask = y_true != 0
    mare = np.mean(np.abs((y_true[mask] - pred[mask]) / y_true[mask])) * 100
    print(f"\n[검증] 평균 상대 오차(MARE): {mare:.2f}%")

    OUT = "rul_lstm_all_sets.h5"
    model.save(OUT)
    print("✅  모델 저장 :", OUT)
