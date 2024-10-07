# Moodify

"감정에 기반한 음악 추천 프로젝트"

"자연어 처리(NIP) 활용 프로젝트"

# 데이터 전처리
- 의미가 전달되지 않는 불용어(STOPWORDS) 제거
- 각종 특수문자 및 문장부호 제거
- 공백, 중복 데이터 제거
- 이모지 제거

# 데이터 세트
- AI HUB에서 가져옴
- 감정 정보가 포함된 단발성, 연속성 대화 dataset
- 총 9만개가 넘는 감정 정보 중 일부를 sample로 진행
- 훈련 데이터와 검증 데이터 교차

# 모델 학습
- 전처리된 데이터를 기반으로 진행함
- 하이퍼파라미터 설정(epochs = 10, batch_size = 128)

# 모델 평가
- 초기 Loss : 1.2 ~ 1.5 (7/03)

- 초기 F1 score : 0.4 ~ 0.5 (7/03)
  
- 이후 Loss : 0.1 ~ 0.2 (9/20)

- 이후 F1 score 값 : 0.66(9/20)
  
- 현재 Loss : 0.02 ~ 0.03 (10/07)

- 현재 F1 score 값 : 0.69 ~ 0.70(10/07)


**최종 목표** => F1-Score 0.7 

## 검증 데이터를 교차하여 진행할 것 (과적합 주의)

# 음악 알고리즘 생성
- ~ing
