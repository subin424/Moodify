import re
import pandas as pd
from konlpy.tag import Okt

def pp_text(text):
    if pd.isna(text):
        return ""
    if isinstance(text, float):
        return ""
    # 공백 제거 및 이모지 제거
    clean_text = re.sub(r'\s+', ' ', text)
    clean_text = re.sub(r'[\U00010000-\U0010ffff]', '', clean_text)
    # 특수문자 제거 추가 
    clean_text = re.sub(r'[^가-힣0-9\sㅋㅋㅠㅠㅜㅜㅎㅎ]+', '', clean_text)
    return clean_text.strip()

def tk_text(text):
    okt = Okt()
    tokens = okt.morphs(text)
    return tokens

def rm_stopwords(tokens, stop_words):
    filter_tokens = [token for token in tokens if token not in stop_words and token]
    return filter_tokens

def preprocess_data(df):
    # 중복 데이터 제거
    df = df.drop_duplicates()
    return df

def balance_emotions(df, target_count=200):
    # 각 감정별로 최대 target_count만큼 샘플링
    balanced_df = df.explode('Emotion').groupby('Emotion').apply(
        lambda x: x.sample(n=min(len(x), target_count), random_state=42)
    ).reset_index(drop=True)

    # 부족한 감정이 있을 경우, 모든 감정의 수를 맞추기 위해 부족한 만큼 복제
    all_emotions = balanced_df['Emotion'].unique()
    max_count = balanced_df['Emotion'].value_counts().min()  # 현재의 최소 감정 수

    for emotion in all_emotions:
        count = balanced_df[balanced_df['Emotion'] == emotion].shape[0]
        if count < target_count:
            # 부족한 수 만큼 랜덤으로 복제
            additional_samples = balanced_df[balanced_df['Emotion'] == emotion].sample(
                n=target_count - count, replace=True, random_state=42
            )
            balanced_df = pd.concat([balanced_df, additional_samples], ignore_index=True)

    # 랜덤 샘플로 인한 중복 방지 및 인덱스 리셋
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df

def pp_excel_file(input_file, output_file, sample_size=10000):
    # Excel 파일을 DataFrame으로 읽기
    df = pd.read_excel(input_file)

    # 중복 데이터 제거
    df = preprocess_data(df)

    # 데이터를 랜덤하게 샘플링하여 적은 양의 데이터로 만듭니다.
    sampled_df = df.sample(n=sample_size, random_state=42)

    # 불용어 리스트 불러오기
    with open("C:/Users/Main/Desktop/dataset/stopwords.txt", "r", encoding="utf-8") as file:
        stop_words = [line.strip() for line in file]

    # 필요한 열 선택 (B1: 'Sentence', B2: 'Emotion')
    sampled_df = sampled_df[['Sentence', 'Emotion']]
    sampled_df.columns = ['Sentence', 'Emotion']  # 열 이름 변경

    # 각 열에 대해 텍스트 전처리 적용
    for col in sampled_df.columns:
        sampled_df[col] = sampled_df[col].apply(pp_text)  # 텍스트 전처리
        sampled_df[col] = sampled_df[col].apply(tk_text)  # 토큰화
        sampled_df[col] = sampled_df[col].apply(rm_stopwords, stop_words=stop_words)  # 불용어 제거

    # 감정 데이터 균형 맞추기
    balanced_df = balance_emotions(sampled_df, target_count=4000)

    # 빈 토큰 리스트 제거
    balanced_df = balanced_df[balanced_df['Sentence'].str.len() > 0]  # 'Sentence'에서 비어있는 리스트 제거
    balanced_df = balanced_df[balanced_df['Emotion'].str.len() > 0]  # 'Emotion'에서 비어있는 리스트 제거

    # 전처리된 데이터를 새로운 Excel 파일로 저장
    balanced_df.to_excel(output_file, index=False)

input_excel_file = 'C:/Users/Main/Desktop/dataset/단발성대화데이터.xlsx'
output_excel_file = 'C:/Users/Main/Desktop/dataset/Dataset11.xlsx'

pp_excel_file(input_excel_file, output_excel_file, sample_size=20000) 