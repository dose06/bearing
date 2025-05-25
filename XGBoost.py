# rul_model_trainer_all_sets.py (Spearman 개선: 낮은 threshold + 상위 N개 선택 + 다채널 + 자동 하이퍼파라미터 튜닝)
"""
▶ 추가 전략
--------------------------------------------------
- Spearman+Fault 기반 주파수 선택, 에너지 엔트로피 3배 강조
- XGBoost 회귀 모델로 RUL 예측
- Optuna를 이용한 자동 하이퍼파라미터 튜닝 포함
"""
import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import kurtosis, skew, spearmanr
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import optuna
from nptdms import TdmsFile

# ▶ 공통 유틸리티 함수 (피처 추출 / 데이터 준비)
# (기존 process_all_sets, create_sequences, compute_percent_error, compute_arul_score 여기에 포함)
def load_vibration_data(file_path):
    tdms = TdmsFile.read(file_path)
    grp = tdms.groups()[0].name
    return pd.DataFrame({ch.name: ch.data for ch in tdms[grp].channels()})

def extract_timestamp(f):
    ts = os.path.basename(f).split("_")[-1].replace(".tdms","")
    return pd.to_datetime(ts, format="%Y%m%d%H%M%S")
fault_freqs = [140,93,78]

# Spearman+Fault 주파수 선택
SELECTED_FREQ_INDICES = {}
FREQ_VECTOR = None

def compute_selected_frequency_indices(file_list, channels, top_n=10, fs=25600):
    psd_ch = {ch: [] for ch in channels}
    rul_list = []
    for df, rul in file_list:
        for ch in channels:
            if ch in df.columns:
                f,Pxx = welch(df[ch].values, fs=fs)
                psd_ch[ch].append(Pxx)
        rul_list.append(rul)
    selected = {}
    for ch in channels:
        mat = np.array(psd_ch[ch])
        if mat.size==0: continue
        rho = [abs(spearmanr(mat[:,i], rul_list)[0]) for i in range(mat.shape[1])]
        top = np.argsort(rho)[-top_n:]
        fault_idx = [np.argmin(np.abs(f - ff)) for ff in fault_freqs]
        idx = sorted(set(list(top)+fault_idx))
        selected[ch]=idx
    return selected, f

# 에너지 엔트로피
def energy_entropy_selected(data, idx, fs=25600):
    f,Pxx = welch(data, fs=fs)
    p = Pxx[idx]/Pxx[idx].sum()
    p = p[p>0]
    return -np.sum(p*np.log(p))

# 전체 데이터 처리
def process_all_sets(root, channels=["CH1","CH2","CH3","CH4"], top_n=10):
    global SELECTED_FREQ_INDICES, FREQ_VECTOR
    pairs=[]
    for d in sorted(glob(os.path.join(root, "Train*"))):
        files=sorted(glob(os.path.join(d,"*.tdms")), key=extract_timestamp)
        times=[extract_timestamp(f) for f in files]
        end=max(times)
        for f,t in zip(files[:-1], times[:-1]): pairs.append((load_vibration_data(f),(end-t).total_seconds()))
    SELECTED_FREQ_INDICES, FREQ_VECTOR = compute_selected_frequency_indices(pairs, channels, top_n)
    rows=[]
    for df,rul in pairs:
        for sec in np.array_split(df, len(df)//25600):
            feats={}
            for ch in channels:
                if ch in sec:
                    d=sec[ch].values;f,Pxx=welch(d)
                    feats[f'{ch}_mean']=d.mean(); feats[f'{ch}_std']=d.std()
                    feats[f'{ch}_entropy']=energy_entropy_selected(d, SELECTED_FREQ_INDICES.get(ch,[]))*3
            feats['RUL']=rul; rows.append(feats)
    return pd.DataFrame(rows)

# 시퀀스 생성 및 평탄화
def prepare_xgb_data(df, window=5, test_size=0.1, rs=42):
    scaler=MinMaxScaler(); X=scaler.fit_transform(df.drop(columns=['RUL'])); y=df['RUL'].values
    X_seq=[X[i:i+window] for i in range(len(X)-window+1)]; y_seq=[y[i+window-1] for i in range(len(X)-window+1)]
    X_flat=np.vstack([s.reshape(1,-1) for s in X_seq])
    y_arr=np.array(y_seq)
    bins=[-1,30,300,2000,np.inf]; labels=pd.cut(y_arr,bins,labels=False)
    return train_test_split(X_flat, y_arr, test_size=test_size, random_state=rs, stratify=labels)

# 평가 지표
def compute_percent_error(act,pred):
    act,pred=np.array(act),np.array(pred)
    m=act!=0; err=np.zeros_like(act); err[m]=100*(act[m]-pred[m])/act[m]; return err

def compute_arul_score(eri):
    return np.where(eri<=0, np.exp(-np.log(0.5)*eri/20), np.exp(np.log(0.5)*eri/50))

# Optuna 튜닝
DATA_ROOT = r"c:/Users/조성찬/OneDrive - UOS/바탕 화면/배어링데이터"
WINDOW=5
def objective(trial):
    df=process_all_sets(DATA_ROOT)
    X_tr,X_val,y_tr,y_val=prepare_xgb_data(df,WINDOW)
    params={
        'learning_rate':trial.suggest_loguniform('learning_rate',0.01,0.3),
        'max_depth':trial.suggest_int('max_depth',4,12),
        'n_estimators':trial.suggest_int('n_estimators',100,500),
        'subsample':trial.suggest_uniform('subsample',0.6,1.0),
        'colsample_bytree':trial.suggest_uniform('colsample_bytree',0.6,1.0),
        'random_state':42,'n_jobs':-1
    }
    model=XGBRegressor(**params)
    model.fit(X_tr,y_tr)
    pred=model.predict(X_val)
    eri=compute_percent_error(y_val,pred)
    return np.mean(np.abs(eri))

if __name__=='__main__':
    study=optuna.create_study(direction='minimize')
    study.optimize(objective,n_trials=20)
    print('Best MARE:',study.best_value)
    print('Params:',study.best_params)
    # 최적 모델 학습 및 평가
    df=process_all_sets(DATA_ROOT)
    X_tr,X_val,y_tr,y_val=prepare_xgb_data(df,WINDOW)
    best=XGBRegressor(**study.best_params,random_state=42,n_jobs=-1)
    best.fit(X_tr,y_tr)
    pred=best.predict(X_val)
    mae=mean_absolute_error(y_val,pred)
    eri=compute_percent_error(y_val,pred)
    mare=np.mean(np.abs(eri))
    a_rul=np.mean(compute_arul_score(eri))
    print(f"MAE: {mae:.2f}, MARE: {mare:.2f}%, A_RUL: {a_rul:.4f}")
    best.save_model('tuned_rul_xgb.json')
