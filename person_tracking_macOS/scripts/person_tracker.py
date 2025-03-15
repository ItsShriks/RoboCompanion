import cv2
import mediapipe as mp
import numpy as np

class PersonFollow:
    def __init__(self):
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        
        # Configuration parameters
        self.FRAME_WIDTH = 640
        self.CENTER_THRESHOLD = 100
        self.TARGET_DISTANCE = 1.0
        self.MOVEMENT_THRESHOLD = 50
        self.speed = 0.2
        self.angular_speed = 0.08
        
        # State variables
        self.person_detected = False
        self.current_distance = None
        self.last_depth = None

        # Initialize Video Capture
        self.cap = cv2.VideoCapture(0)  # Open the default webcam

    def is_stop_gesture(self, landmarks):
        tips = [landmarks.landmark[i] for i in [8, 12, 16, 20]]
        base = landmarks.landmark[0]
        return all([tip.y > base.y for tip in tips])

    def process_frame(self, frame):
        try:
            # Convert OpenCV image to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            results_hands = self.hands.process(rgb_frame)

            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    if self.is_stop_gesture(hand_landmarks):
                        print("Stop gesture detected, stopping the robot")
                        return 'succeeded'

            if results.pose_landmarks:
                self.person_detected = True
                h, w, _ = frame.shape
                landmarks = results.pose_landmarks.landmark

                x_coords = [int(landmark.x * w) for landmark in landmarks]
                y_coords = [int(landmark.y * h) for landmark in landmarks]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                x = int(nose.x * w)

                center_x = w // 2
                movement_direction = "CENTER"

                if x < center_x - self.CENTER_THRESHOLD:
                    movement_direction = "LEFT"
                    self.move_left(self.angular_speed)
                elif x > center_x + self.CENTER_THRESHOLD:
                    movement_direction = "RIGHT"
                    self.move_right(self.angular_speed)
                else:
                    self.move_straight(self.speed)

                print(f"Movement Direction: {movement_direction}")

            else:
                self.person_detected = False

            cv2.imshow("Person Tracker", frame)

        except Exception as e:
            print(f"Error processing frame: {str(e)}")

    def move_left(self, speed):
        print("Moving Left")

    def move_right(self, speed):
        print("Moving Right")

    def move_straight(self, speed):
        print("Moving Straight")

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.process_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    person_follower = PersonFollow()
    person_follower.run()