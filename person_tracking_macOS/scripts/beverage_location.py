import cv2
import requests
import numpy as np

# Replace with your model details
ROBOFLOW_MODEL = "graduate-thesis/2"
API_KEY = "5tuCnshmsOMCKOYAVSuP"
URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}?api_key={API_KEY}"

def get_object_position(frame_width, bbox):
    """Determine if an object is in LEFT, CENTER, or RIGHT."""
    if bbox is None:
        return "NONE"
    
    x, y, w, h = bbox  # Bounding box (x, y, width, height)
    object_center = x + w // 2  # Get the center x-coordinate
    
    left_threshold = frame_width // 3
    right_threshold = 2 * (frame_width // 3)
    
    if object_center < left_threshold:
        return "LEFT"
    elif object_center > right_threshold:
        return "RIGHT"
    else:
        return "CENTRE"

def process_frame(frame, target_class):
    """Send frame to Roboflow for object detection and process results."""
    _, img_encoded = cv2.imencode(".jpg", frame)
    response = requests.post(URL, files={"file": img_encoded.tobytes()})
    
    if response.status_code != 200:
        print("Error:", response.text)
        return "NONE", frame

    detections = response.json().get("predictions", [])

    frame_width = frame.shape[1]
    position = "NONE"

    for det in detections:
        x, y, w, h = int(det["x"] - det["width"] / 2), int(det["y"] - det["height"] / 2), int(det["width"]), int(det["height"])
        detected_class = det["class"]

        if detected_class == target_class:
            position = get_object_position(frame_width, (x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{detected_class} ({position})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return position, frame

def main():
    cap = cv2.VideoCapture(0)
    target_class = input("Enter the class name to locate: ").strip()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        position, frame = process_frame(frame, target_class)

        cv2.putText(frame, f"{target_class} Position: {position}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()