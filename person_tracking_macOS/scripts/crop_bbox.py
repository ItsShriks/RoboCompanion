import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from ultralytics import YOLO
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Initialize ROS node
rospy.init_node('person_cropper', anonymous=True)

# Load YOLO model for person detection
model = YOLO('yolov8n.pt')  # Use a lightweight YOLO model

# Initialize CV bridge
bridge = CvBridge()

# ROS Publisher for bounding boxes in RViz
bbox_pub = rospy.Publisher("/detected_person/bounding_boxes", Marker, queue_size=10)

# Global variables to store image data
rgb_image = None
depth_image = None
bounding_box = None  # Store the detected bounding box

def publish_bounding_box(x1, y1, x2, y2):
    """Publish the bounding box as a Marker for RViz."""
    marker = Marker()
    marker.header.frame_id = "camera_link"  # Adjust according to your frame
    marker.header.stamp = rospy.Time.now()
    marker.ns = "person_detection"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.02  # Line width

    marker.color.r = 1.0  # Red color
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0  # Fully visible

    marker.points = [
        Point(x1, y1, 0), Point(x2, y1, 0),
        Point(x2, y2, 0), Point(x1, y2, 0),
        Point(x1, y1, 0)  # Close the bounding box
    ]

    bbox_pub.publish(marker)

def rgb_callback(msg):
    global rgb_image, bounding_box

    try:
        # Convert ROS image to OpenCV format
        rgb_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Run person detection
        results = model(rgb_image)

        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:  # Class 0 is 'person' in COCO dataset
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bounding_box = (x1, y1, x2, y2)

                    # Publish bounding box for visualization in RViz
                    publish_bounding_box(x1, y1, x2, y2)

                    # Crop the bounding box from the RGB image
                    cropped_person = rgb_image[y1:y2, x1:x2]

                    # Display cropped image
                    cv2.imshow("Cropped RGB Person", cropped_person)
                    cv2.waitKey(1)
                    return  # Process only one person

    except Exception as e:
        print("RGB Error:", e)

def depth_callback(msg):
    global depth_image, bounding_box

    try:
        # Convert ROS depth image to OpenCV format
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        if bounding_box is not None:
            x1, y1, x2, y2 = bounding_box

            # Crop the depth image using the same bounding box
            cropped_depth = depth_image[y1:y2, x1:x2]

            # Normalize depth for visualization
            depth_display = cv2.normalize(cropped_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = np.uint8(depth_display)

            # Show cropped depth image
            cv2.imshow("Cropped Depth Person", depth_display)
            cv2.waitKey(1)

    except Exception as e:
        print("Depth Error:", e)

# Subscribe to XTion camera topics
rospy.Subscriber("/camera/rgb/image_raw", Image, rgb_callback)
rospy.Subscriber("/camera/depth/image_raw", Image, depth_callback)

rospy.spin()