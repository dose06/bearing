import os
import pandas as pd
from nptdms import TdmsFile
import datetime

# ▶ 파일 경로
file_path = r"c:\Users\조성찬\OneDrive - UOS\바탕 화면\배어링데이터\Validation2\modified_KIMM Simulator_KIMM Bearing Test_20160422055414.tdms"

# ▶ 저장 경로
save_dir = r"c:\Users\조성찬\OneDrive - UOS\바탕 화면\배어링데이터\csv"
os.makedirs(save_dir, exist_ok=True)

# ▶ 파일 이름에서 날짜 추출 함수
def extract_time_from_filename(file_path):
    base = os.path.basename(file_path)
    try:
        time_str = base.split("_")[-1].replace(".tdms", "")  # 예: 20160402141507
        dt = datetime.datetime.strptime(time_str, "%Y%m%d%H%M%S")
        return dt.strftime("%Y%m%d_%H%M%S")  # → '20160402_141507'
    except:
        return "unknown_time"

# ▶ TDMS 로딩 함수
def load_tdms_file(file_path):
    tdms_file = TdmsFile.read(file_path)
    group_name_vibration = tdms_file.groups()[0].name
    group_name_operation = tdms_file.groups()[1].name

    vib_channels = tdms_file[group_name_vibration].channels()
    vib_data = {ch.name: ch.data for ch in vib_channels}

    operation_channels = tdms_file[group_name_operation].channels()
    operation_data = {ch.name: ch.data for ch in operation_channels}

    return vib_data, operation_data

import numpy as np
from scipy.signal import welch
from nptdms import TdmsFile

def load_signal(file_path, channel='CH1'):
    """
    TDMS 파일에서 지정 채널의 1차원 진동 데이터 배열을 반환합니다.
    """
    tdms = TdmsFile.read(file_path)
    # 첫 번째 그룹에서 데이터프레임으로 변환
    group = tdms.groups()[0]
    df = tdms[group.name].as_dataframe()
    return df[channel].values

def entropy_of_window(signal, fs, win_size):
    """
    signal: 1차원 진동 데이터 (np.ndarray)
    fs: 샘플링 주파수 (Hz)
    win_size: 윈도우 길이 (초)
    """
    N = int(fs * win_size)
    # 윈도우 구간 PSD 계산
    f, Pxx = welch(signal[:N], fs=fs)
    p = Pxx / np.sum(Pxx)
    p = p[p > 0]
    # 엔트로피 공식
    return -np.sum(p * np.log(p))

if __name__ == "__main__":
    # 사용자 설정
    fs = 25600  # 샘플링 주파수
    file_path = r"c:\Users\조성찬\OneDrive - UOS\바탕 화면\배어링데이터\Validation2\modified_KIMM Simulator_KIMM Bearing Test_20160422055414.tdms"
    channel   = "CH1"

    # 신호 불러오기
    signal = load_signal(file_path, channel)

    # 윈도우 크기별 엔트로피 계산
    window_sizes = [0.1, 0.2, 0.5, 1.0]  # 초 단위
    entropies = {}
    for ws in window_sizes:
        entropies[ws] = entropy_of_window(signal, fs, ws)

    # 결과 출력
    for ws, H in entropies.items():
        print(f"Window {ws:0.1f}s ({int(fs*ws)} samples): Entropy = {H:.4f}")


# ▶ 데이터 로딩
vib_data, operation_data = load_tdms_file(file_path)
vib_df = pd.DataFrame(vib_data)
operation_df = pd.DataFrame(operation_data)

# ▶ 시간 문자열 붙여서 저장
time_str = extract_time_from_filename(file_path)
vib_csv_path = os.path.join(save_dir, f"vibration_{time_str}.csv")
operation_csv_path = os.path.join(save_dir, f"operation_{time_str}.csv")

vib_df.to_csv(vib_csv_path, index=False)
operation_df.to_csv(operation_csv_path, index=False)

print(f"✅ 진동 데이터 저장 완료: {vib_csv_path}")
print(f"✅ 운전 데이터 저장 완료: {operation_csv_path}")
