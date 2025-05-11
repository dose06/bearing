import os
import pandas as pd
from nptdms import TdmsFile
import datetime

# ▶ 파일 경로
file_path = r"c:\Users\조성찬\OneDrive - UOS\바탕 화면\배어링데이터\Train7\modified_KIMM Simulator_KIMM Bearing Test_20160529075522.tdms"

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
