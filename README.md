# Moodify: 감정 기반 음악 추천 시스템

**Moodify**는 카카오톡 대화 내용을 분석하여 감정을 추출하고, 그에 맞는 음악을 유튜브에서 추천해주는 시스템입니다. KoELECTRA 모델을 사용하여 감정을 분류하고, 각 감정에 맞는 음악을 추천합니다.

## 프로젝트 개요

- **모델**: 이 프로젝트는 [KoELECTRA](https://github.com/monologg/KoELECTRA) 모델을 사용하여 감정을 분류합니다. 분류된 감정은 기쁨, 슬픔, 중립, 두려움, 분노, 혐오, 놀람으로 구분합니다.
- **음악 추천 알고리즘**: 감정 분석 결과에 따라, 각 감정에 맞는 유튜브 플레이리스트를 추천합니다.
- **사용자 인터페이스**: Tkinter 라이브러리를 사용한 간단한 그래픽 사용자 인터페이스(GUI)를 통해 카카오톡 대화 파일을 업로드하고 감정 분석을 수행하며, 음악 추천 결과를 제공합니다.

## 주요 기능

1. **감정 분석**: 카카오톡 대화 파일에서 텍스트를 추출한 후 KoELECTRA 모델을 사용하여 감정을 분류합니다. 지원하는 감정은 다음과 같습니다:
   - 기쁨 (Happy)
   - 슬픔 (Sad)
   - 중립 (Neutral)
   - 두려움 (Fear)
   - 분노 (Anger)
   - 혐오 (Disgust)
   - 놀람 (Surprise)

2. **음악 추천**: 감정 분석 결과에 따라 해당 감정에 맞는 유튜브 플레이리스트를 추천하고, 브라우저에서 자동으로 열립니다.

3. **진행 상태 표시**: 대화를 분석하는 동안 진행 상태를 보여주는 프로그레스 바가 GUI에 포함되어 있어 분석 진행 상황을 확인할 수 있습니다.

## 동작 방식

1. **모델 및 토크나이저 로드**: 미리 학습된 KoELECTRA 모델과 토크나이저를 지정된 디렉터리에서 로드합니다.
   
2. **텍스트 추출**: 카카오톡 대화 파일에서 텍스트 데이터를 추출하고 분석을 위한 전처리를 수행합니다.

3. **감정 분석**: 대화 내용을 KoELECTRA 모델을 사용하여 감정 분류를 진행하고, 가장 많이 등장한 감정을 분석합니다.

4. **감정 기반 음악 추천**: 분석된 주요 감정에 맞는 유튜브 플레이리스트를 브라우저에서 실행합니다.

1. 데이터 전처리
- 의미가 전달되지 않는 불용어(STOPWORDS) 제거
- 각종 특수문자 및 문장부호 제거
- 공백, 중복 데이터 제거
- 이모지 제거
- 각 감정의 수 균형 맞춤

# 데이터 세트
- AI HUB에서 가져옴
- 감정 정보가 포함된 단발성, 연속성 대화 dataset
- 총 9만개가 넘는 감정 정보 중 일부를 sample로 진행
- 훈련 데이터와 검증 데이터 교차
- 총 7가지의 감정 (행복, 슬픔, 중립, 공포, 분노, 혐오, 놀람)

# 사용한 모델
- Koelectra

- <https://github.com/monologg/KoELECTRA>

# 모델 학습

- 전처리한 데이터 기반으로 진행함
- 하이퍼파라미터 기본 설정(평균적으로 epochs = 5-10, batch_size = 128 lr = 2e-5)
=> 더 나은 결과를 얻도록 계속해서 바꿔가며 진행
- 검증 데이터를 교차하여 진행 (과적합 주의)

# 모델 평가

- 초기 Loss : 1.2 ~ 1.5 (7/03)
- 초기 F1 score : 0.4 ~ 0.5 (7/03)

- 이후 Loss : 0.1 ~ 0.2 (9/20)
- 이후 F1 score 값 : 0.66 (9/20)
  
- 현재 Loss : 0.02 ~ 0.03 (10/07)
- 현재 F1 score 값 : 0.69 ~ 0.70 (10/07)


*최종 목표* => F1-Score 0.7 - 0.75


# 모델 학습 Log

 <https://github.com/subin424/Moodify/blob/main/Project/Model/README.md>

# 음악 알고리즘 생성
- 모델
