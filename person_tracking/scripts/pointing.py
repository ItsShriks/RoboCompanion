from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO models
pose_model = YOLO("yolov8n-pose.pt")  # For pose detection
object_model = YOLO("yolov8n.pt")      # For object detection (includes bottles)

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
        
        # Detect objects (bottles)
        object_results = object_model(frame)
        
        # Extract bottle detections
        bottles = []
        if len(object_results) > 0 and hasattr(object_results[0], 'boxes') and hasattr(object_results[0].boxes, 'data'):
            for det in object_results[0].boxes.data:
                if len(det) >= 6:  # Ensure we have enough values (x1,y1,x2,y2,conf,cls)
                    x1, y1, x2, y2, conf, cls = det
                    # Class 39 is 'bottle' in COCO dataset
                    if int(cls) == 39 and conf > 0.5:
                        bottles.append((int(x1), int(y1), int(x2), int(y2)))
                        # Draw bottle boxes in blue
                        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        # Detect poses
        pose_results = pose_model(frame)
        
        # Track which bottles are being pointed at
        pointed_bottles = set()
        
        # Flags for detected hands
        right_hand_detected = False
        left_hand_detected = False
        
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
                        if person_keypoints.shape[0] <= 10:
                            continue  # Skip if not enough keypoints
                        
                        # Process LEFT arm (inverse, so use right arm indices)
                        if np.all(person_keypoints[7][:2] > 0) and np.all(person_keypoints[9][:2] > 0):
                            # Get elbow and wrist points
                            elbow = person_keypoints[7][:2].astype(int)
                            wrist = person_keypoints[9][:2].astype(int)
                            
                            # Find intersection point
                            intersection = extend_line_get_endpoint(wrist, elbow, width, height)
                            
                            # Determine pointing direction
                            pointing_direction = "RIGHT" if intersection[0] < wrist[0] else "LEFT"
                            
                            # Draw the line
                            cv2.line(display_frame, tuple(wrist), intersection, (0, 0, 255), 2)
                            
                            # Add text indicating hand and direction at the end of the line
                            text_pos = (
                                intersection[0] - 90 if intersection[0] > 90 else intersection[0] + 10,
                                intersection[1] - 10 if intersection[1] > 10 else intersection[1] + 20
                            )
                            cv2.putText(display_frame, f"LEFT → {pointing_direction}", text_pos, 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            # Set the left hand flag
                            left_hand_detected = True
                            
                            # Check if line intersects with any bottle
                            for i, (bx1, by1, bx2, by2) in enumerate(bottles):
                                if line_intersects_box(wrist, intersection, (bx1, by1, bx2, by2)):
                                    pointed_bottles.add(i)
                        
                        # Process RIGHT arm (inverse, so use left arm indices)
                        if np.all(person_keypoints[8][:2] > 0) and np.all(person_keypoints[10][:2] > 0):
                            # Get elbow and wrist points
                            elbow = person_keypoints[8][:2].astype(int)
                            wrist = person_keypoints[10][:2].astype(int)
                            
                            # Find intersection point
                            intersection = extend_line_get_endpoint(wrist, elbow, width, height)
                            
                            # Determine pointing direction
                            pointing_direction = "RIGHT" if intersection[0] < wrist[0] else "LEFT"
                            
                            # Draw the line
                            cv2.line(display_frame, tuple(wrist), intersection, (0, 0, 255), 2)
                            
                            # Add text indicating hand and direction at the end of the line
                            text_pos = (
                                intersection[0] - 90 if intersection[0] > 90 else intersection[0] + 10,
                                intersection[1] - 10 if intersection[1] > 10 else intersection[1] + 20
                            )
                            cv2.putText(display_frame, f"RIGHT → {pointing_direction}", text_pos, 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            # Set the right hand flag
                            right_hand_detected = True
                            
                            # Check if line intersects with any bottle
                            for i, (bx1, by1, bx2, by2) in enumerate(bottles):
                                if line_intersects_box(wrist, intersection, (bx1, by1, bx2, by2)):
                                    pointed_bottles.add(i)
        
        # Display hand detection status
        status_text = []
        if right_hand_detected:
            status_text.append("Right hand detected")
        if left_hand_detected:
            status_text.append("Left hand detected")
        
        if status_text:
            cv2.putText(display_frame, " & ".join(status_text), (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Highlight pointed bottles in green
        for i in pointed_bottles:
            if i < len(bottles):
                bx1, by1, bx2, by2 = bottles[i]
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
                cv2.putText(display_frame, "POINTED", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Pose Detection & Bottle Pointing', display_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def extend_line_get_endpoint(point2, point1, width, height):
    """
    Extends a line from point1 through point2 until it reaches the image boundary.
    Returns the endpoint at the boundary.
    """
    # Convert points to integers
    x1, y1 = point1
    x2, y2 = point2
    
    # Calculate the direction vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Handle case where points are the same
    if dx == 0 and dy == 0:
        return (x2, y2)
    
    # Normalize the direction vector
    magnitude = np.sqrt(dx**2 + dy**2)
    dx /= magnitude
    dy /= magnitude
    
    # Find parameter t for intersection with image boundaries
    t_values = []
    
    # Left boundary (x = 0)
    if dx < 0:
        t_values.append(-x2 / dx)
    
    # Right boundary (x = width-1)
    if dx > 0:
        t_values.append((width-1 - x2) / dx)
    
    # Top boundary (y = 0)
    if dy < 0:
        t_values.append(-y2 / dy)
    
    # Bottom boundary (y = height-1)
    if dy > 0:
        t_values.append((height-1 - y2) / dy)
    
    # Get the minimum positive t value
    valid_t_values = [t for t in t_values if t >= 0]
    if valid_t_values:
        t = min(valid_t_values)
        
        # Calculate the end point
        x_end = int(x2 + dx * t)
        y_end = int(y2 + dy * t)
        
        return (x_end, y_end)
    else:
        return (x2, y2)

def line_intersects_box(line_start, line_end, box):
    """
    Determines if a line intersects with a bounding box.
    box is in format (x1, y1, x2, y2)
    """
    # Unpack the box coordinates
    x1, y1, x2, y2 = box
    
    # Convert line to parametric form
    x1_line, y1_line = line_start
    x2_line, y2_line = line_end
    
    # Check if either endpoint is inside the box
    if (x1 <= x1_line <= x2 and y1 <= y1_line <= y2) or (x1 <= x2_line <= x2 and y1 <= y2_line <= y2):
        return True
    
    # Check if line intersects any of the box edges
    edges = [
        ((x1, y1), (x2, y1)),  # Top edge
        ((x2, y1), (x2, y2)),  # Right edge
        ((x2, y2), (x1, y2)),  # Bottom edge
        ((x1, y2), (x1, y1))   # Left edge
    ]
    
    for edge_start, edge_end in edges:
        if line_segments_intersect(line_start, line_end, edge_start, edge_end):
            return True
    
    return False

def line_segments_intersect(p1, p2, p3, p4):
    """
    Check if two line segments (p1, p2) and (p3, p4) intersect.
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

if __name__ == "__main__":
    capture_webcam()
