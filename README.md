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
GRU+attention

위의 순서로 발전시켜왔습니다.

.cvs 파일은 데이터 확인용입니다

노션정리 주소
[https://ambiguous-origami-76c.notion.site/1d59d01c0fbe806ba862cf6062a856ac?pvs=4](https://ambiguous-origami-76c.notion.site/1d59d01c0fbe806ba862cf6062a856ac?pvs=4)
