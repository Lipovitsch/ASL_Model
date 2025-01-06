import cv2
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
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
            keypoints.extend([lm.x, lm.y, lm.z])  # Dodaj współrzędne x, y, z
        return np.array(keypoints).reshape(1, -1)
    return None

# Funkcja do przewidywania klasy
def predict_class(keypoints):
    predictions = model.predict(keypoints, verbose=0)
    predicted_label = classes[np.argmax(predictions)]
    if np.max(predictions) < 0.9:
        predicted_label = "Nie wykryto litery."
    return predicted_label

# Funkcja do aktualizacji obrazu w czasie rzeczywistym
def update_frame():
    global current_frame
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)  # Odbicie lustrzane
        frame, results = process_frame(frame)  # Przetwarzanie klatki
        current_frame = (frame, results)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        lbl_video.imgtk = img
        lbl_video.configure(image=img)
        lbl_video.after(10, update_frame)

# Funkcja do przechwytywania obrazu i przewidywania klasy
def capture_and_predict():
    global current_frame
    frame, results = current_frame
    keypoints = extract_keypoints(results)
    if keypoints is not None:
        # Przewidywanie klasy
        predicted_label = predict_class(keypoints)

        # Wyświetlenie wyciętej dłoni
        lbl_result.config(text=f"Przewidywana litera: {predicted_label}")

        # Wyświetlenie przechwyconej dłoni
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        lbl_hand.imgtk = img
        lbl_hand.configure(image=img)
    else:
        lbl_result.config(text="Nie wykryto dłoni.")

# Konfiguracja GUI
root = Tk()
root.title("ASL Model")

# Obraz z kamerki
lbl_video = Label(root)
lbl_video.pack()

# Przycisk do przechwytywania obrazu
btn_capture = Button(root, text="Zrób zdjęcie", command=capture_and_predict)
btn_capture.pack()

# Etykieta wyników
lbl_result = Label(root, text="Przewidywana litera: ---")
lbl_result.pack()

# Obraz przechwyconej dłoni
lbl_hand = Label(root)
lbl_hand.pack()

# Konfiguracja kamery
cap = cv2.VideoCapture(0)
current_frame = None
update_frame()

# Uruchomienie aplikacji
root.mainloop()
cap.release()
cv2.destroyAllWindows()
