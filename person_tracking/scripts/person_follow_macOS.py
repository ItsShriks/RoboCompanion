import cv2
import mediapipe as mp
import torch
import torchreid
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize TorchReID model for person re-identification
torchreid.models.show_avai_models()
reid_model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1, pretrained=True)
reid_model.eval()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reid_model.to(device)

# Open webcam
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to RGB (MediaPipe requires RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Extract key points of the body
        landmarks = results.pose_landmarks.landmark

        # Use the midpoint between shoulders (landmarks 11 & 12) as a reference
        shoulder_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x +
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2 * frame_width
        
        frame_center = frame_width // 2

        # Determine person's position
        if shoulder_x < frame_center * 0.4:
            direction = "LEFT"
        elif shoulder_x > frame_center * 0.6:
            direction = "RIGHT"
        else:
            direction = "STRAIGHT"

        # Draw direction text on the frame
        cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw pose landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Person Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()