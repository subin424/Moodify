import pandas as pd
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import webbrowser
import torch
from googleapiclient.discovery import build
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# 모델과 토크나이저 로드
model_path = 'C:/Users/Main/Desktop/result'
tokenizer = ElectraTokenizer.from_pretrained(model_path)
model = ElectraForSequenceClassification.from_pretrained(model_path)

# 감정 기반 노래 추천
emotion_song_mapping = {
    'happy': [
        'https://www.youtube.com/watch?v=m1DLUwkqHlU&list=PL3lFOT1N1evop8NTowvUxbtOhbKCvDju9&index=18'],
    'sad': [
        'https://www.youtube.com/watch?v=uc3dpkLTYzQ'],
    'neutral': [
        'https://www.youtube.com/watch?v=BMv6MNg3Vqs'],
    'fear': [
        'https://www.youtube.com/watch?v=V7JZGBz6ZOc'],
    'anger': [
        'https://www.youtube.com/watch?v=nNhm3Z5ylqw'],
    'disgust': [
        'https://www.youtube.com/watch?v=Mk9Cn-ii6N4'],
    'surprise': [
        'https://www.youtube.com/watch?v=ZQWrleq-NhQ']
}

# 카카오톡 대화 파일에서 텍스트 추출
def extract_text_from_kakao(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.readlines()
    return [text.strip() for text in text_data]

# 감정 분석
def analyze_emotion(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    emotions = ['happy', 'sad', 'neutral', 'fear', 'anger', 'disgust', 'surprise']
    return emotions[predicted_class]

# 유튜브 플레이리스트 생성
def create_youtube_playlist(emotion):
    songs = emotion_song_mapping.get(emotion, [])
    playlist = []
    for song in songs:
        webbrowser.open(song)
        playlist.append(song)
    return playlist

# main 함수
def main(kakao_file):
    texts = extract_text_from_kakao(kakao_file)
    emotion_count = {'happy': 0, 'sad': 0, 'neutral': 0, 'fear': 0, 'anger': 0, 'disgust': 0, 'surprise': 0}

    # 프로그레스 바 초기화
    progress['maximum'] = len(texts)

    for text in texts:
        emotion = analyze_emotion(text)
        emotion_count[emotion] += 1
        progress['value'] += 1  # 프로그레스 바 업데이트
        root.update_idletasks()  # 프로그레스 바 갱신

    # 감정 분석 결과에서 가장 많이 나온 감정 선택
    dominant_emotion = max(emotion_count, key=emotion_count.get)

    # 해당 감정에 맞는 유튜브 플레이리스트 생성
    playlist = create_youtube_playlist(dominant_emotion)

    # 결과를 메시지 박스로 표시
    messagebox.showinfo("결과", f" 감정 : {dominant_emotion}\nYouTube Playlist:\n" + "\n".join(playlist))

# GUI 초기화
root = tk.Tk()
root.title("Moodify")

# 프로그레스 바 설정
progress = ttk.Progressbar(root, length=300, mode='determinate')
progress.pack(pady=20)

# 버튼 클릭 시 메인 함수 실행
button = tk.Button(root, text="분석 시작", command=lambda: main(r'C:\Users\Main\Desktop\dataset\moodify3.txt'))
button.pack(pady=10)

# GUI 실행
root.mainloop()