import cv2
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
import mediapipe as mp

# Załaduj model
model = load_model(r"asl_model_handpoints.keras")

# Załaduj klasy
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
           'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']

# Konfiguracja MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Zmienna przechowująca aktualnie zbudowane słowo
current_text = ""

# Licznik klatek do ograniczenia częstotliwości przewidywań
frame_counter = 0
predict_every_n_frames = 30  # Przewiduj co 15 klatek

# Funkcja do przetwarzania klatek
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
    return frame, results

# Funkcja do ekstrakcji kluczowych punktów
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints).reshape(1, -1)
    return None

# Funkcja do przewidywania klasy
def predict_class(keypoints):
    predictions = model.predict(keypoints, verbose=0)
    confidence = np.max(predictions)
    predicted_label = classes[np.argmax(predictions)]
    if confidence < 0.8:
        return None
    return predicted_label

# Aktualizacja klatki i przewidywanie litery
def update_frame():
    global current_text, frame_counter

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame_with_landmarks, results = process_frame(frame)

        frame_counter += 1
        if frame_counter >= predict_every_n_frames:
            frame_counter = 0
            keypoints = extract_keypoints(results)
            if keypoints is not None:
                predicted_label = predict_class(keypoints)
                if predicted_label:
                    if predicted_label == 'space':
                        current_text += ' '
                    elif predicted_label == 'del':
                        current_text = current_text[:-1]
                    else:
                        current_text += predicted_label
                    lbl_result.config(text=f"Wykryty tekst: {current_text}")

        frame_rgb = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        lbl_video.imgtk = img
        lbl_video.configure(image=img)

    lbl_video.after(10, update_frame)

# Resetowanie tekstu
def reset_text():
    global current_text
    current_text = ""
    lbl_result.config(text="Wykryty tekst: ")

# Konfiguracja GUI
root = Tk()
root.title("ASL Model")

# Obraz z kamerki
lbl_video = Label(root)
lbl_video.pack()

# Etykieta wynikowego tekstu
lbl_result = Label(root, text="Wykryty tekst: ", font=("Helvetica", 16))
lbl_result.pack(pady=10)

# Przycisk resetujący tekst
btn_reset = Button(root, text="Resetuj tekst", command=reset_text)
btn_reset.pack(pady=5)

# Konfiguracja kamery
cap = cv2.VideoCapture(0)
update_frame()

# Uruchomienie aplikacji
root.mainloop()
cap.release()
cv2.destroyAllWindows()
