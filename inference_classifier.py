
import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
from threading import Thread
import pyttsx3
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from twilio.rest import Client

# Constants
MODEL_PATH = 'C:/Users/prakh/OneDrive/Desktop/Hand Sign2/sign-language-detector-python-master/model.p'
OUTPUT_FILE_PATH = "predicted_characters.txt"
CAMERA_INDICES = [0, 1, 2, 3]
LABELS_DICT = {0: 'EMERGENCY HELP', 1: 'I NEED MEDICAL HELP', 2: 'CRIMINAL BEHIND ME', 3: 'FIRE FIRE',
               4: 'ANIMAL ATTACK'}
UPDATE_INTERVAL = 500  # milliseconds

# Email Configuration
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECIPIENT_ADDRESS = os.getenv("RECIPIENT_ADDRESS")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

TWILIO_ACCOUNT_SID = "VA7e7f98e9bc517b36ee3b021817d59378"
TWILIO_AUTH_TOKEN = "5GU2JKXD2GCQX6ZELJWCVYKJ"
TWILIO_PHONE_NUMBER = "+917905213399"
RECIPIENT_PHONE_NUMBER = "+916397659166"


def send_email(file_path):
    try:
        message = MIMEMultipart()
        message['From'] = EMAIL_ADDRESS
        message['To'] = RECIPIENT_ADDRESS
        message['Subject'] = "Predicted Characters File"

        part = MIMEBase('application', "octet-stream")
        with open(file_path, "rb") as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(file_path)}"')

        message.attach(part)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, RECIPIENT_ADDRESS, message.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")


def send_sms_message(message):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    try:
        sms_message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )
        print(f"SMS sent. SID: {sms_message.sid}")
    except Exception as e:
        print(f"Failed to send SMS: {str(e)}")


class RealTimeDisplay(tk.Tk):
    def __init__(self, file_path):
        super().__init__()
        self.title("Real-Time Predictions")
        self.geometry("400x200")

        self.text_widget = tk.Text(self, wrap='word')
        self.text_widget.pack(expand=True, fill='both')
        self.file_path = file_path
        self.update_text()

    def update_text(self):
        try:
            with open(self.file_path, 'r') as file:
                self.text_widget.delete('1.0', tk.END)
                self.text_widget.insert(tk.END, file.read())
        except Exception as e:
            self.text_widget.insert(tk.END, str(e))
        self.after(UPDATE_INTERVAL, self.update_text)  # Update every second


def try_camera_indices(indices):
    for index in indices:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def capture_predictions():
    model_dict = pickle.load(open(MODEL_PATH, 'rb'))
    model = model_dict['model']

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    capture = try_camera_indices(CAMERA_INDICES)
    if not capture:
        print("Unable to access any camera")
        return

    engine = pyttsx3.init()

    with open(OUTPUT_FILE_PATH, "w") as output_file:
        while True:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = capture.read()
            if not ret:
                print("Failed to capture image")
                break

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = LABELS_DICT[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
                output_file.write(predicted_character + '\n')  # Make sure each entry is on a new line
                output_file.flush()
                engine.say(predicted_character)
                engine.runAndWait()

                # Triggering the SMS message
                send_sms_message(predicted_character)

                time.sleep(2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()
    print(EMAIL_ADDRESS, EMAIL_PASSWORD, RECIPIENT_ADDRESS)

    # Send the email after capturing predictions
    send_email(OUTPUT_FILE_PATH)


if __name__ == "__main__":
    # Start the capture in a separate thread
    capture_thread = Thread(target=capture_predictions)
    capture_thread.start()

    # Start the Tkinter GUI
    app = RealTimeDisplay(OUTPUT_FILE_PATH)
    app.mainloop()
    capture_thread.join()

