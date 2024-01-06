import cv2
import mediapipe as mp
import pyautogui
from datetime import datetime

# Initialize mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Get screen width and height
screen_width, screen_height = pyautogui.size()

# Open the webcam (you might need to change the index based on your setup)
cap = cv2.VideoCapture(0)

# Variables for click detection
last_blink_time = datetime.now()
blink_threshold = 1.5  # Adjust this threshold based on your preference
double_click_threshold = 0.5  # Adjust this threshold based on your preference
last_click_time = datetime.now()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with mediapipe FaceMesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Extract the coordinates of the left eye (you might need to adjust the indices)
            eye_x = int(landmarks.landmark[159].x * frame.shape[1])
            eye_y = int(landmarks.landmark[159].y * frame.shape[0])

            # Map the eye coordinates to the screen coordinates
            target_x = int(eye_x / frame.shape[1] * screen_width)
            target_y = int(eye_y / frame.shape[0] * screen_height)

            # Draw a circle around the left eye
            cv2.circle(frame, (eye_x, eye_y), 10, (0, 255, 0), -1)

            # Move the cursor to the mapped coordinates
            pyautogui.moveTo(target_x, target_y)

            # Check for a blink (left-click)
            blink = landmarks.landmark[159].y < landmarks.landmark[145].y

            if blink:
                # Check if enough time has passed since the last blink
                current_time = datetime.now()
                time_since_last_blink = (current_time - last_blink_time).total_seconds()

                if time_since_last_blink > blink_threshold:
                    # Perform left-click action
                    pyautogui.click()

                    # Update the last blink time
                    last_blink_time = current_time

                    # Check for a double-click
                    time_since_last_click = (current_time - last_click_time).total_seconds()

                    if time_since_last_click < double_click_threshold:
                        # Perform double-click action
                        pyautogui.click()

                    # Update the last click time
                    last_click_time = current_time

    # Display the frame
    cv2.imshow("Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()