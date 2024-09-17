import re
import pandas as pd
from konlpy.tag import Okt

def pp_text(text):
    if pd.isna(text):
        return ""
    if isinstance(text, float):
        return ""
    clean_text = re.sub(r'\s+', ' ', text)
    clean_text = re.sub(r'[\U00010000-\U0010ffff]', '', clean_text)
    return clean_text.strip()

def tk_text(text):
    okt = Okt()
    tokens = okt.morphs(text)
    return tokens

def rm_stopwords(tokens, stop_words):
    filter_tokens = [token for token in tokens if token not in stop_words]
    return filter_tokens

def preprocess_data(df):
    # 중복 데이터 제거
    df = df.drop_duplicates()
    return df

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

    # 각 열에 대해 텍스트 전처리 적용
    for col in sampled_df.columns:
        sampled_df[col] = sampled_df[col].apply(pp_text)
        sampled_df[col] = sampled_df[col].apply(tk_text)
        sampled_df[col] = sampled_df[col].apply(rm_stopwords, stop_words=stop_words)

    # 전처리된 데이터를 새로운 Excel 파일로 저장
    sampled_df.to_excel(output_file, index=False)

input_excel_file = 'C:/Users/Main/Desktop/dataset/연속성대화데이터.xlsx'
output_excel_file = 'C:/Users/Main/Desktop/dataset/Dataset.xlsx'

pp_excel_file(input_excel_file, output_excel_file, sample_size=50000)