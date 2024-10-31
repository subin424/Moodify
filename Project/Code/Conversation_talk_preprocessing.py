import re
import pandas as pd
from konlpy.tag import Okt

# 제거할 특정 단어 리스트
remove_words = ['오전', '오후', '수빈', '한준']

def pp_text(text):
    if pd.isna(text):
        return ""
    if isinstance(text, float):
        return ""
    # 공백 제거 및 이모지 제거
    clean_text = re.sub(r'\s+', ' ', text)
    clean_text = re.sub(r'[\U00010000-\U0010ffff]', '', clean_text)
    # 숫자 제거
    clean_text = re.sub(r'\d+', '', clean_text)
    # 특수문자 및 불필요한 문자 제거
    clean_text = re.sub(r'[^가-힣\sㅋㅋㅠㅠㅜㅜㅎㅎ]+', '', clean_text)
    return clean_text.strip()

def tk_text(text):
    okt = Okt()
    tokens = okt.morphs(text)
    return tokens

def rm_stopwords(tokens, stop_words):
    filter_tokens = [token for token in tokens if token not in stop_words and token]
    return filter_tokens

def remove_duplicates(tokens):
    seen = set()
    unique_tokens = []
    for token in tokens:
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)
    return unique_tokens  # 중복 제거된 토큰 리스트 반환

def remove_specific_words(tokens):
    return [token for token in tokens if token not in remove_words]  # 특정 단어 제거

def preprocess_data(df):
    # 중복 데이터 제거
    df = df.drop_duplicates()
    return df

def pp_text_file(input_file, output_file, sample_size=10000):
    # 텍스트 파일을 DataFrame으로 읽기
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # DataFrame으로 변환
    df = pd.DataFrame(lines, columns=['Sentence'])

    # 중복 데이터 제거
    df = preprocess_data(df)

    # 샘플링할 수의 최대 크기 조정
    max_sample_size = min(sample_size, len(df))

    # 데이터를 랜덤하게 샘플링하여 적은 양의 데이터로 만듭니다.
    sampled_df = df.sample(n=max_sample_size, random_state=42)

    # 불용어 리스트 불러오기
    with open("C:/Users/Main/Desktop/dataset/stopwords.txt", "r", encoding="utf-8") as file:
        stop_words = [line.strip() for line in file]

    # 각 열에 대해 텍스트 전처리 적용
    sampled_df['Processed'] = sampled_df['Sentence'].apply(pp_text)  # 텍스트 전처리
    sampled_df['Tokens'] = sampled_df['Processed'].apply(tk_text)  # 토큰화
    sampled_df['Tokens'] = sampled_df['Tokens'].apply(rm_stopwords, stop_words=stop_words)  # 불용어 제거
    sampled_df['Tokens'] = sampled_df['Tokens'].apply(remove_duplicates)  # 반복되는 단어 제거
    sampled_df['Tokens'] = sampled_df['Tokens'].apply(remove_specific_words)  # 특정 단어 제거

    # 빈 토큰 리스트 제거
    sampled_df = sampled_df[sampled_df['Tokens'].str.len() > 0]  # 'Tokens'에서 비어있는 리스트 제거

    # 전처리된 데이터를 새로운 텍스트 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for tokens in sampled_df['Tokens']:
            f.write(' '.join(tokens) + '\n')

input_text_file = 'C:/Users/Main/Desktop/dataset/moodify.txt'  # 입력 텍스트 파일 경로
output_text_file = 'C:/Users/Main/Desktop/dataset/Processed_moodify.txt'  # 출력 텍스트 파일 경로

pp_text_file(input_text_file, output_text_file, sample_size=20000)