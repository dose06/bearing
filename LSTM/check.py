import matplotlib.pyplot as plt
from nptdms import TdmsFile

file_path = r"c:\Users\조성찬\OneDrive - UOS\바탕 화면\배어링데이터\Train1\modified_KIMM Simulator_KIMM Bearing Test_20160326035839.tdms"
tdms_file = TdmsFile.read(file_path)

from nptdms import TdmsFile
import pandas as pd

def load_tdms_file(file_path):
    tdms_file = TdmsFile.read(file_path)

    # group_name_vibration: {CH1, CH2, CH3, CH4} 데이터, group_name_operation: {Torque[Nm], TC SP Front[℃], TC SP Rear[℃]} 데이터
    group_name_vibration = tdms_file.groups()[0].name
    group_name_operation = tdms_file.groups()[1].name

    vib_channels = tdms_file[group_name_vibration].channels()
    vib_data = {ch.name: ch.data for ch in vib_channels}
    
    operation_channels = tdms_file[group_name_operation].channels()
    operation_data = {ch.name: ch.data for ch in operation_channels}

    return vib_data , operation_data 

vib_data, operation_data = load_tdms_file(file_path)

# 👉 데이터프레임으로 변환하여 출력
vib_df = pd.DataFrame(vib_data)
operation_df = pd.DataFrame(operation_data)

print("🔹 진동 데이터 (Vibration):")
print(vib_df.head())  # 상위 5개만 출력

print("\n🔹 운전 데이터 (Operation):")
print(operation_df.head())
import os

save_dir = r"C:\Users\조성찬\OneDrive - UOS\바탕화면\배어링데이터"
os.makedirs(save_dir, exist_ok=True)

vib_df.to_csv(os.path.join(save_dir, "vibration_data.csv"), index=False)
operation_df.to_csv(os.path.join(save_dir, "operation_data.csv"), index=False)