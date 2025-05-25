import os
import matplotlib.pyplot as plt
import pandas as pd
from nptdms import TdmsFile
import datetime
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates

# 날짜 포맷: 월-일 시:분
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.gcf().autofmt_xdate()  # 자동 회전

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 음수 기호 깨짐 방지


DATA_ROOT = r"c:\Users\조성찬\OneDrive - UOS\바탕 화면\배어링데이터"

def extract_timestamp(filepath):
    name = os.path.basename(filepath)
    time_str = name.split("_")[-1].replace(".tdms", "")
    return datetime.datetime.strptime(time_str, "%Y%m%d%H%M%S")

def load_tdms_basic(file_path):
    tdms_file = TdmsFile.read(file_path)
    vib_group = tdms_file.groups()[0].name
    op_group = tdms_file.groups()[1].name

    vib_data = {ch.name: ch.data for ch in tdms_file[vib_group].channels()}
    op_data = {ch.name: ch.data for ch in tdms_file[op_group].channels()}
    return pd.DataFrame(vib_data), pd.DataFrame(op_data)

def compute_rms(vib_df):
    return vib_df.pow(2).mean(axis=0).pow(0.5).mean()  # 전체 채널 평균 RMS

def get_matching_column(df, keyword):
    for col in df.columns:
        if keyword in col:
            return col
    raise KeyError(f"'{keyword}'를 포함하는 열을 찾을 수 없습니다: {df.columns.tolist()}")

set_folders = sorted([f for f in glob(os.path.join(DATA_ROOT, "Train*")) if os.path.isdir(f)])

for set_path in set_folders:
    rms_list = []
    torque_list = []
    temp_front = []
    temp_rear = []
    timestamps = []

    tdms_files = sorted(glob(os.path.join(set_path, "*.tdms")), key=extract_timestamp)

    for file_path in tdms_files:
        try:
            vib_df, op_df = load_tdms_basic(file_path)
            rms = compute_rms(vib_df)

            torque_col = get_matching_column(op_df, "Torque")
            temp_col = get_matching_column(op_df, "Front")
            temp_rear_col = get_matching_column(op_df, "Rear")

            torque = op_df[torque_col].mean()
            temp_f = op_df[temp_col].mean()
            temp_r = op_df[temp_rear_col].mean()

            timestamps.append(extract_timestamp(file_path))
            rms_list.append(rms)
            torque_list.append(torque)
            temp_front.append(temp_f)
            temp_rear.append(temp_r)  # ✅ 여기서 append

        except Exception as e:
            print(f"❌ {file_path} 오류: {e}")

    # ▶ 그래프 그리기 (Torque를 오른쪽 축에)
    plt.figure(figsize=(12, 6))
    plt.title(f"{os.path.basename(set_path)}: 열화 및 운동 특성")
    plt.plot(timestamps, temp_front, label="Temp Front [℃]", color='green')
    plt.plot(timestamps, temp_rear, label="Temp Rear [℃]", color='orange')
    
    ax = plt.gca()
    ax2 = ax.twinx()  # 오른쪽 y축
    ax2.plot(timestamps, torque_list, label="Torque [Nm]", color='red', linestyle='--')

    # ✅ 날짜 포맷 설정
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [℃]")
    ax2.set_ylabel("Torque [Nm]")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.grid(True)
    plt.tight_layout()
    plt.show()
