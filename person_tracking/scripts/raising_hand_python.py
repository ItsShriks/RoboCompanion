from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model for pose detection
pose_model = YOLO("yolov8n-pose.pt")

def capture_webcam():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam opened successfully. Press 'q' to quit.")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break

        # Get dimensions
        height, width = frame.shape[:2]
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Detect poses
        pose_results = pose_model(frame)
        
        # Process each detected person
        for result in pose_results:
            # Check if keypoints exist and are not None
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                # Check if there's any keypoint data
                if hasattr(result.keypoints, 'data') and result.keypoints.data is not None:
                    # Convert keypoints to numpy array
                    try:
                        keypoints = result.keypoints.data.cpu().numpy()
                    except:
                        continue  # Skip if conversion fails
                    
                    # Skip if keypoints array is empty
                    if keypoints.size == 0:
                        continue
                    
                    for person_keypoints in keypoints:
                        # Check if we have enough keypoints
                        if person_keypoints.shape[0] <= 16:  # Need head and hands
                            continue
                        
                        # Get nose position (keypoint 0) for head location
                        if np.all(person_keypoints[0][:2] > 0):
                            nose = person_keypoints[0][:2].astype(int)
                            
                            # Check if either hand is raised
                            hand_raised = False
                            
                            # Check left wrist (keypoint 10) is above the nose
                            if np.all(person_keypoints[10][:2] > 0):
                                left_wrist = person_keypoints[10][:2].astype(int)
                                if left_wrist[1] < nose[1]:  # y-coordinate is smaller (higher in image)
                                    hand_raised = True
                            
                            # Check right wrist (keypoint 9) is above the nose
                            if np.all(person_keypoints[9][:2] > 0):
                                right_wrist = person_keypoints[9][:2].astype(int)
                                if right_wrist[1] < nose[1]:  # y-coordinate is smaller (higher in image)
                                    hand_raised = True
                            
                            # If hand is raised, draw an inverted triangle above the head
                            if hand_raised:
                                # Triangle top point is 50 pixels above the nose
                                triangle_top = (nose[0], nose[1] - 50)
                                # Triangle base is 30 pixels wide
                                triangle_left = (nose[0] - 15, nose[1] - 20)
                                triangle_right = (nose[0] + 15, nose[1] - 20)
                                
                                # Draw filled inverted triangle
                                triangle_pts = np.array([triangle_top, triangle_left, triangle_right], np.int32)
                                triangle_pts = triangle_pts.reshape((-1, 1, 2))
                                cv2.fillPoly(display_frame, [triangle_pts], (0, 255, 0))
                                
                                # Add a text label
                                cv2.putText(display_frame, "HAND RAISED", 
                                           (nose[0] - 50, nose[1] - 60),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Hand Raising Detection', display_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_webcam()