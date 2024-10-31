import pandas as pd
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import webbrowser
import torch
from tkinter import messagebox
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk

# 모델과 토크나이저 로드
model_path = 'C:/Users/Main/Desktop/result'
tokenizer = ElectraTokenizer.from_pretrained(model_path)
model = ElectraForSequenceClassification.from_pretrained(model_path)

# 유튜브 플레이리스트 생성
emotion_song_mapping = {
    'happy': ['https://www.youtube.com/watch?v=m1DLUwkqHlU&list=PL3lFOT1N1evop8NTowvUxbtOhbKCvDju9&index=18'],
    'sad': ['https://youtu.be/uc3dpkLTYzQ?si=JCeNrxHP73kJwhfI'],
    'neutral': ['https://www.youtube.com/watch?v=BMv6MNg3Vqs'],
    'fear': ['https://www.youtube.com/watch?v=V7JZGBz6ZOc'],
    'anger': ['https://www.youtube.com/watch?v=nNhm3Z5ylqw'],
    'disgust': ['https://www.youtube.com/watch?v=Mk9Cn-ii6N4'],
    'surprise': ['https://www.youtube.com/watch?v=ZQWrleq-NhQ']
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
        # 시작 시간을 0초로 설정하고 자동 재생 및 전체화면 파라미터 추가
        full_screen_url = f"{song.split('&')[0]}?start=0&autoplay=1&fs=1"
        webbrowser.open(full_screen_url)  # 전체화면으로 열기
        playlist.append(full_screen_url)
    return playlist

# 메인 함수
def main(kakao_file):
    texts = extract_text_from_kakao(kakao_file)
    emotion_count = {emotion: 0 for emotion in emotion_song_mapping.keys()}

    # 프로그레스 바 초기화
    progress['maximum'] = len(texts)

    for text in texts:
        emotion = analyze_emotion(text)
        emotion_count[emotion] += 1
        progress['value'] += 1  # 프로그레스 바 업데이트
        root.update_idletasks()  # 프로그레스 바 갱신

    # 감정 분석 결과에서 가장 많이 나온 감정 선택
    dominant_emotion = max(emotion_count, key=emotion_count.get)

    # 해당 감정과 일치한 유튜브 플레이리스트 생성
    playlist = create_youtube_playlist(dominant_emotion)

    # 결과를 메시지 박스로 표시
    messagebox.showinfo("결과", f"감정: {dominant_emotion}\nYouTube Playlist:\n" + "\n".join(playlist))

# GUI 초기화
root = tk.Tk()
root.title("Moodify")
root.attributes('-fullscreen', True)  # 전체 화면 설정

# 'Esc' 키로 전체 화면 종료
root.bind("<Escape>", lambda e: root.attributes('-fullscreen', False))

# 배경 이미지 설정
background_image_path = "C:/Users/Main/Downloads/background.jpg"  # 여기에 배경 이미지 파일 경로 입력
bg_image = Image.open(background_image_path)
bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

# 배경 이미지를 담을 Canvas 생성
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# 중앙 정렬을 위한 프레임 생성 (위쪽으로 이동)
center_frame = tk.Frame(root, bg="#f0f0f0", bd=2, relief="solid")
center_frame.place(relx=0.5, rely=0.2, anchor="center")  # rely 값을 0.2로 조정하여 약간 위로 이동

# 타이틀 라벨
title_label = tk.Label(center_frame, text="Moodify 플레이리스트 ",
                       font=("Arial", 24, "bold"), bg="#f0f0f0", fg="#333333")
title_label.pack(pady=20)

# 텍스트 박스 (작게 조정 및 꾸미기)
text_box = tk.Text(center_frame, wrap='word', height=4, width=40, bg="#EAEAEA", fg="#333333",
                   borderwidth=2, relief="groove", font=("Arial", 12))
text_box.pack(pady=10)

# 프로그레스 바
progress = ttk.Progressbar(center_frame, length=600, mode='determinate')
progress.pack(pady=30)

# 분석 시작 버튼
button = tk.Button(center_frame, text="분석 시작", command=lambda: main(r'C:\Users\Main\Desktop\dataset\moodify1.txt'),
                   font=("Arial", 16), bg="#4CAF50", fg="white")
button.pack(pady=30)

# GUI 실행
root.mainloop()
