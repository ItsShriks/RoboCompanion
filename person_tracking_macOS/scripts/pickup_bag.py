import torch
import cv2
import numpy as np
import torch.serialization

# Add the custom class to the safe globals list
torch.serialization.add_safe_globals([torch.nn.Module])  # or include the class that caused the issue if known

# Load your custom model
model = torch.load("/Users/shrikar/Library/Mobile Documents/com~apple~CloudDocs/Sem III/b-it bots @Home/RoboCompanion/person_tracking/scripts/best_shopping_bag.pt", weights_only=False)
model.eval()

# Open the webcam
cap = cv2.VideoCapture(0)

def detect_shopping_bag(frame):
    # Convert the frame from a numpy array to a torch tensor
    frame_tensor = torch.from_numpy(frame).float()

    # If the frame is a color image (3 channels), we need to adjust the shape
    if frame_tensor.ndimension() == 3:
        frame_tensor = frame_tensor.permute(2, 0, 1)  # Change from (H, W, C) to (C, H, W)

    # Normalize the frame to the range [0, 1] (if necessary)
    frame_tensor /= 255.0

    # Ensure the frame is on the right device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_tensor = frame_tensor.to(device)

    # Add a batch dimension
    frame_tensor = frame_tensor.unsqueeze(0)

    # Run the model on the frame
    with torch.no_grad():
        results = model(frame_tensor)

    return results

def draw_bounding_box_and_handle(frame, results):
    # Get the results, typically results.xywh[0] will contain bounding boxes
    bbox = results.xywh[0].cpu().numpy()
    for box in bbox:
        x_center, y_center, width, height, confidence, class_id = box
        if confidence > 0.5:  # Adjust confidence threshold if needed
            x1, y1, x2, y2 = int(x_center - width / 2), int(y_center - height / 2), int(x_center + width / 2), int(y_center + height / 2)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate the coordinates for the handle (upper center of the bounding box)
            handle_x, handle_y = int(x_center), int(y1)  # Handle coordinates at the top center
            cv2.circle(frame, (handle_x, handle_y), 5, (0, 0, 255), -1)

            # Display the coordinates on the frame
            cv2.putText(frame, f"Bag Handle: ({handle_x}, {handle_y})", (handle_x + 10, handle_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return frame

def main():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for better performance if needed
        frame = cv2.resize(frame, (640, 480))

        # Detect shopping bag and handle
        results = detect_shopping_bag(frame)

        # Draw bounding box and handle coordinates
        frame_with_bboxes = draw_bounding_box_and_handle(frame, results)

        # Display the resulting frame
        cv2.imshow('Shopping Bag Detection', frame_with_bboxes)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()