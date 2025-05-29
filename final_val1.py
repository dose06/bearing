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


# rul_model_trainer_all_sets.py (Spearman 개선: 낮은 threshold + 상위 N개 선택 + 다채널)
'''
채널 4개(CH1 ~ CH4) → 어떻게 처리하고 결합했는가?

### 결합 방식

각 채널에서 추출한 특징들을 **하나의 벡터**로 **병렬 결합(concatenate)**합니다.

CH1_mean, CH1_std, CH1_entropy, ..., CH4_band_power, CH4_entropy

—>>>>>>그렇다면 엔트로피의 비중이 낮으므로 키우자(엔트로피값 *3)
->>>>>> ch별 상위 10개 추출
->>>>>> 초단위 라벨링
->>>>>> 고장 주파수 반영
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


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼 공백 제거 + ℃ → °C 통일"""
    df.columns = [col.strip().replace("℃", "°C") for col in df.columns]
    return df

def load_vibration_data(file_path: str) -> pd.DataFrame:
    """
    TDMS 파일 하나로부터 진동 데이터(Vibration) + 운전 데이터(Operation)을 병합한 DataFrame 반환
    """
    tdms_file = TdmsFile.read(file_path)
    group_map = {g.name: g for g in tdms_file.groups()}

    # ─ 1. 진동 데이터 추출 ─
    if "Vibration" not in group_map:
        raise ValueError(f"❌ 'Vibration' 그룹이 존재하지 않음: {file_path}")
    
    vib_group = group_map["Vibration"]
    vib_data = {
        ch.name.strip(): ch[:] for ch in vib_group.channels()
        if ch.name.strip().startswith("CH")
    }
    vib_df = pd.DataFrame(vib_data)
    vib_df = _normalize_cols(vib_df)

    # ─ 2. 운전 데이터 병합 (Operation 그룹) ─
    if "Operation" in group_map:
        op_group = group_map["Operation"]
        for ch in op_group.channels():
            col_name = ch.name.strip().replace("℃", "°C")
            if len(ch) > 0:
                vib_df[col_name] = ch[0]
            else:
                vib_df[col_name] = np.nan
    else:
        print(f"⚠️ Operation 그룹이 없습니다: {file_path}")
        for col in ['Torque[Nm]', 'TC SP Front[°C]', 'TC SP Rear[°C]']:
            vib_df[col] = np.nan

    return vib_df

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



def iso_life_loss_from_torque(
    torque_series,
    shaft_radius_m = 0.015,         # 축 반경 [m]
    Fa_N = 15_000.0,                # 고정 축방향 하중 [N]
    C_dyn_N = 59_500.0,             # 동적 정격 하중 [N]
    rpm_ref = 1000,                 # 회전 속도 [rpm]
    X_coef = 0.56, Y_coef = 1.5     # 테이퍼 베어링용 ISO 계수
):
    """
    ISO 281 기반 수명 소모율 계산 (토크 기반 반경하중 + 고정 축방향 하중)
    
    Parameters
    ----------
    torque_series : np.ndarray or pd.Series
        1초간 토크 [N·m]
    shaft_radius_m : float
        회전축 반경 [m] (예: 25 mm)
    Fa_N : float
        축방향 하중 [N] (정격값 고정)
    C_dyn_N : float
        동적 정격 하중 [N] (예: 59.5 kN)
    rpm_ref : int
        회전 속도 [rpm]
    X_coef, Y_coef : float
        ISO 281의 하중 계수

    Returns
    -------
    float
        iso_life_loss : 1초당 수명 소모율 (0~1 사이 소수)
    """

    # ① 평균 토크 → 반경방향 하중 [N]
    torque_mean = np.mean(np.abs(torque_series))
    Fr_N = torque_mean / shaft_radius_m

    # ② 등가 하중 계산 (ISO 기준)
    P_N = X_coef * Fr_N + Y_coef * Fa_N
    P_N = max(P_N, 1.0)  # 안정성 확보 (0 나눗셈 방지)

    # ③ ISO 281 수명 계산 (백만 회전 단위)
    L10_revs = (C_dyn_N / P_N) ** (10 / 3) * 1_000_000

    # ④ 회전수를 시간으로 환산 → 수명 [초]
    rev_per_sec = rpm_ref / 60
    life_sec = L10_revs / rev_per_sec

    # ⑤ 1초 슬라이스니까, 전체 수명 중 1초만큼 닳음
    return float(1.0 / life_sec)

def compute_ETCI_inverse_group(entropy_vals, temp_celsius):
    T_K = temp_celsius + 273.15
    mean_entropy = np.mean([max(e, 1e-6) for e in entropy_vals])
    return (1.0 / mean_entropy) * np.log(T_K)

def compute_TWP_group(Pxx_list, temp_celsius, T_ref=30.0):
    T_ratio = temp_celsius / T_ref
    total_power = np.sum([np.sum(pxx) if isinstance(pxx, np.ndarray) else pxx for pxx in Pxx_list])
    return float(total_power * T_ratio)

# ▶ 특징 추출 함수
def extract_features_from_vibration(vib_df, sampling_rate=25600):
    """
    1 초 슬라이스 DataFrame → 특징 딕셔너리 1 개 반환
    (진동 CHx + 운전 채널 mean/std + 추가 통계·PSD 특징 포함)
    """
    features = {}
    Pxx_dict = {}
    entropy_dict = {}
    # -------- 진동 채널(CH1~) --------
    psd_done = False  # PSD 통계 한 번만 계산 플래그

    for ch in vib_df.columns:
        if not ch.startswith('CH'):
            continue  # 운전 채널 건너뜀

        data = vib_df[ch].values
        rms  = np.sqrt(np.mean(data**2))
        Imax, Imin = data.max(), data.min()
        Ip        = 0.5 * (Imax - Imin)
        Iamean    = np.mean(np.abs(data))

        # ── 기본 통계 (이미 있던 것) ──
        features[f'{ch}_mean']       = float(data.mean())
        features[f'{ch}_std']        = float(data.std())
        features[f'{ch}_rms']        = float(rms)
        features[f'{ch}_kurtosis']   = float(kurtosis(data))
        features[f'{ch}_skew']       = float(skew(data))
        features[f'{ch}_crest']      = float(np.abs(Imax) / rms)
        
        # ── 신규: 충격·리플 특성 ──
        features[f'{ch}_impulse']    = float(Ip / Iamean)   # Iimpulse
        features[f'{ch}_p2rms']      = float(np.abs(Imax) / rms)  # Ip2rms

        # ── 신규: 분포 강건형 ──
        features[f'{ch}_median']     = float(np.median(data))
        q25, q75 = np.percentile(data, [25, 75])
        features[f'{ch}_perc25']     = float(q25)
        features[f'{ch}_perc75']     = float(q75)
        features[f'{ch}_iqr']        = float(q75 - q25)

        # ── 신규: 에너지 분산 ──
        features[f'{ch}_sigma2']     = float(np.var(data))

        # ── PSD 계산 (모든 채널 공통) ──
        _, Pxx = welch(data, fs=sampling_rate)
        features[f'{ch}_band_power'] = float(Pxx.sum())
        Pxx_dict[ch] = Pxx  # ✅ Pxx 저장
        
        # 엔트로피 (선택 주파수만)
        if ch in SELECTED_FREQ_INDICES:
            ent = energy_entropy_selected(
                data, SELECTED_FREQ_INDICES[ch], sampling_rate)
            features[f'{ch}_entropy'] = float(ent * 3)  # 가중치 ×3
            entropy_dict[ch] = ent  # ✅ 엔트로피 저장
        # ---------- PSD 통계는 채널 1번만 ----------
        if not psd_done:
            psd_done = True
            features['Fmax'] = float(Pxx.max())
            features['Frms'] = float(np.sqrt(np.mean(Pxx**2)))
            features['Fvar'] = float(np.var(Pxx))

    # -------- 운전 채널 mean/std --------
    for op in ['Torque[Nm]', 'TC SP Front[°C]', 'TC SP Rear[°C]']:
        if op in vib_df.columns:
            arr = vib_df[op].values
            features[f'{op}_mean'] = float(arr.mean())
            features[f'{op}_std']  = float(arr.std())

    # -------- 토크·온도 상호작용 --------
    # ── (1) 토크와 각속도를 이용한 출력(Power) 계산 ─────────────────────────
    if 'Torque[Nm]' in vib_df.columns:
        
        torque = vib_df['Torque[Nm]'].values
        omega = 2 * np.pi * 1000 / 60             # 1000 rpm → 각속도 [rad/s]
        power = np.mean(torque) * omega           # 평균 토크 × 각속도 → 출력 [Watt]

        # ── (2-A) CHLI_front: 전방 온도를 기반으로 한 열 부하 지표 ────────────────
        if 'TC SP Front[°C]' in vib_df.columns:
            front_temp = vib_df['TC SP Front[°C]'].values
            kelvin_front = front_temp + 273.15                    # 섭씨 → 켈빈 변환
            chli_front = power / np.mean(kelvin_front)            # 출력 / 평균 온도
            features['CHLI_front'] = float(chli_front)            # 최종 피처 저장

        # ── (2-B) CHLI_rear: 후방 온도를 기반으로 한 열 부하 지표 ─────────────────
        if 'TC SP Rear[°C]' in vib_df.columns:
            rear_temp = vib_df['TC SP Rear[°C]'].values
            kelvin_rear = rear_temp + 273.15                      # 섭씨 → 켈빈 변환
            chli_rear = power / np.mean(kelvin_rear)              # 출력 / 평균 온도
            features['CHLI_rear'] = float(chli_rear)              # 최종 피처 저장
             

    # -------- 온도 130 °C 초과 플래그 --------
    if 'TC SP Front[°C]' in vib_df.columns:
        temp_f = vib_df['TC SP Front[°C]'].mean()
        features['front_over130'] = float(np.maximum(0, (temp_f - 130))**2)

    if 'TC SP Rear[°C]' in vib_df.columns:
        temp_r = vib_df['TC SP Rear[°C]'].mean()
        features['rear_over130'] = float(np.maximum(0, (temp_r - 130))**2)

    
    
    if 'Torque[Nm]' in vib_df.columns:
        torque_series = vib_df['Torque[Nm]'].values
        features['iso_life_loss'] = iso_life_loss_from_torque(torque_series) * 1000000000 -600


         # --- FRONT/REAR ETCI + TWP 계산 ---
    temp_front = vib_df['TC SP Front[°C]'].mean() if 'TC SP Front[°C]' in vib_df.columns else 30.0
    temp_rear  = vib_df['TC SP Rear[°C]'].mean()  if 'TC SP Rear[°C]'  in vib_df.columns else 30.0



    features['front_etci'] = compute_ETCI_inverse_group(
        [entropy_dict.get('CH1', 1e-6), entropy_dict.get('CH2', 1e-6)],
        temp_front
    )
    features['rear_etci'] = compute_ETCI_inverse_group(
        [entropy_dict.get('CH3', 1e-6), entropy_dict.get('CH4', 1e-6)],
        temp_rear
    )

        
    T_Kf = temp_front + 273.15
    T_Kr = temp_rear + 273.15

    if 'CH1' in entropy_dict and 'CH2' in entropy_dict and 'CH1' in Pxx_dict and 'CH2' in Pxx_dict:
        S_f = np.mean([entropy_dict['CH1'], entropy_dict['CH2']])
        H_f = np.mean([Pxx_dict['CH1'].sum(), Pxx_dict['CH2'].sum()])
        features['front_gibbs'] = H_f - T_Kf * S_f

    if 'CH3' in entropy_dict and 'CH4' in entropy_dict and 'CH3' in Pxx_dict and 'CH4' in Pxx_dict:
        S_r = np.mean([entropy_dict['CH3'], entropy_dict['CH4']])
        H_r = np.mean([Pxx_dict['CH3'].sum(), Pxx_dict['CH4'].sum()])
        features['rear_gibbs'] = H_r - T_Kr * S_r

 
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





WINDOW = 5                               # 시퀀스 길이


DATA_ROOT = r"c:/Users/조성찬/OneDrive - UOS/바탕 화면/배어링데이터"
MODEL_PATH = 'rul_final3_all_sets1.h5'

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

    offset_base = 29
    for i in range(5):
        start = -(offset_base + WINDOW + i)
        end   = - (offset_base + i)
        seq   = X_all[start:end]
        if len(seq) != WINDOW:
            print(f"Validation{vid} (offset {i}, +1800): 시퀀스 길이 {len(seq)}로 SKIP")
            continue
        pred = model.predict(seq.reshape(1, WINDOW, -1), verbose=0)[0, 0] + 1800
        preds.append(pred)
        print(f"Validation{vid} (offset {i}, +1800): {pred:.1f}")
    
    offset_base = 39
    for i in range(5):
        start = -(offset_base + WINDOW + i)
        end   = - (offset_base + i)
        seq   = X_all[start:end]
        if len(seq) != WINDOW:
            print(f"Validation{vid} (offset {i}, +2400): 시퀀스 길이 {len(seq)}로 SKIP")
            continue
        pred = model.predict(seq.reshape(1, WINDOW, -1), verbose=0)[0, 0] + 2400
        preds.append(pred)
        print(f"Validation{vid} (offset {i}, +2400): {pred:.1f}")


    # preds에 10개의 예측값이 담긴 뒤

    # (1) 전체 평균
    avg_all = sum(preds) / len(preds)

    # (2) 최소·최대 제외한 평균
    sorted_preds = sorted(preds)
    trimmed_preds = sorted_preds[1:-2]            # 첫·마지막(최소·최대) 제거
    avg_trimmed = sum(trimmed_preds) / len(trimmed_preds)

    print(f"전체 {len(preds)}개 평균: {avg_all:.1f}")
    print(f"최소·최대 제외한 {len(trimmed_preds)}개 평균: {avg_trimmed:.1f}")
