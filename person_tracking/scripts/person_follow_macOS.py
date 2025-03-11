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

# Store detected person embeddings
known_person_embeddings = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to RGB (MediaPipe requires RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get shoulder coordinates
        left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]
        right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]

        # Get hip midpoint coordinates (for more stable tracking)
        left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * frame.shape[1]
        right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * frame.shape[1]
        hip_mid_x = (left_hip_x + right_hip_x) / 2  # Midpoint between hips

        frame_center = frame.shape[1] // 2  # Center of the frame

        # Detect if the person is facing away (back view)
        if left_shoulder_x > right_shoulder_x:  # Left shoulder appears right of right shoulder
            view = "BACK"

            # Extract person’s bounding box
            person_x_min = min(left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x)
            person_x_max = max(left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x)
            person_y_min = min(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame.shape[0],
                               landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * frame.shape[0])
            person_y_max = max(landmarks[mp_pose.PoseLandmark.NOSE].y * frame.shape[0],
                               landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * frame.shape[0])

            person_bbox = frame[int(person_y_min):int(person_y_max), int(person_x_min):int(person_x_max)]

            if person_bbox.shape[0] > 0 and person_bbox.shape[1] > 0:
                # Convert to Torch tensor
                person_img = cv2.resize(person_bbox, (128, 256))
                person_tensor = torch.tensor(person_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

                # Extract features using TorchReID
                with torch.no_grad():
                    person_embedding = reid_model(person_tensor)

                # First detection: Store the first detected person’s features
                if known_person_embeddings is None:
                    known_person_embeddings = person_embedding
                else:
                    # Compute similarity
                    similarity = torch.nn.functional.cosine_similarity(known_person_embeddings, person_embedding)

                    if similarity.item() < 0.7:  # If similarity is low, assume it's a different person
                        continue  # Ignore this detection

            # Determine movement direction
            if hip_mid_x < frame_center * 0.9:
                direction = "LEFT"
            elif hip_mid_x > frame_center * 1.1:
                direction = "RIGHT"
            else:
                direction = "STRAIGHT"

            # Display result
            text = f"{direction}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw pose landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Back Tracking with ReID", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()