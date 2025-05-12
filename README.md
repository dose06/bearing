특징 결합 방식
각 채널에서 추출한 특징들을 하나의 벡터로 병렬 결합(concatenate)
예시: CH2_mean, CH2_std, CH2_entropy, ..., CH4_band_power, CH4_entropy

엔트로피 비중 강화
엔트로피 특징에 채널별 가중치 곱셈 (CH2, CH3, CH4 모두 가중치 4) >>>> 모든채널 살리는 방향이 나아보임

Spearman 상관 기반 주파수 선택
각 채널에서 상위 20개 + 고장 관련 주파수 포함

 Welch 스펙트럼 분석 설정
 f, Pxx = welch(data, fs=25600, nperseg=4096) : 주파수 분해능 향상

 RUL 예측 개선을 위한 전략 3가지 적용
 1. RUL log 변환 (log1p) → 모델 안정화
 2. 손실 함수 weighted MAE → 작은 RUL 강조
 3. 예측값 클리핑 (min=10) → MARE 튀는 현상 방지


모델변화 
GRU+attention: 단순 LSTM에 비해 성능이 잘 나오지 않음
-> 데이터 양에 비해서 많은 파라미터 수로 과적합이 유발 되었음
-> 엔트로피 가중치를 파라미터화 시키려면 attention을 사용해야하는거같은데 파라미터가 너무 많아져서 쓰기가 어려울거같음


random seed 를 통해서만 셋을 고정하던걸 hold-out(마지막 파일 데이터를 제외시키므로써 쓰지 않음), Stratified split을 통해서 고정해두니 지금까지와는 다른 비교양상을 보였습니다.
RUL log 변환 (log1p),  손실 함수 weighted MAE 이 상대 오차에는 좋은 영향을 주지만 점수는 낮아짐을 알수있었음
또한 마지막 파일을 훈련 및 평가 셋에 포함하는가 안하는가에 따라서 상대 오차 평균이 많이 달라졌음
지금까지 10초간 측정한 파일의 RUL을 하나로 정하였지만 10초를 1초단위로 나누어 정밀도를 향상시켰고, 결과가 더 좋았음 

점수를 위해서는 RUL log 변환 (log1p),  손실 함수 weighted MAE 를 쓰지 않는것이 좋아보임 

지금 까지 성능이 제일 좋은것은  LSTM_Entropy_spearman4_2.py 
( 10초→ 1초 단위 Labeling & RUL log 변환 (log1p),  손실 함수 weighted MAE 제외 )

노션 정리 주소
https://ambiguous-origami-76c.notion.site/1d59d01c0fbe806ba862cf6062a856ac?pvs=4
