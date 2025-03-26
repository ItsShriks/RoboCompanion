import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os

class PersonDetector:
    def __init__(self, model_path="pose_landmarker_full.task", output_folder="output", num_poses=5, close_threshold=-0.5):
        self.output_folder = output_folder
        self.close_threshold = close_threshold
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            num_poses=num_poses
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def detect_pose(self, image_path):
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        detection_result = self.detector.detect(mp_image)
        return image_bgr, detection_result

    def crop_and_save(self, image_bgr, detection_result):
        """
        Crops, saves, and returns cropped images as a dictionary.

        Returns:
        - cropped_images: Dictionary with keys as part names and values as cropped images.
        """
        cropped_images = {}

        for person_idx, landmarks in enumerate(detection_result.pose_landmarks):
            # avg_z = np.mean([landmark.z for landmark in landmarks])

            # if avg_z > self.close_threshold:
            #     print(f"Skipping person {person_idx}: Too far (avg_z={avg_z:.2f})")
            #     continue

            # print(f"Processing person {person_idx}: Close enough (avg_z={avg_z:.2f})")

            person_crop = self.crop_person(image_bgr, landmarks)
            if person_crop is not None:
                filename = f"{self.output_folder}/person_{person_idx}.jpg"
                cv2.imwrite(filename, person_crop)
                cropped_images[f"person_{person_idx}"] = person_crop
                print(f"Saved {filename}")


        return cropped_images

    @staticmethod
    def crop_person(image, landmarks, expand_ratio=0.1):
        h, w, _ = image.shape
        min_x = int(min([landmark.x for landmark in landmarks]) * w)
        max_x = int(max([landmark.x for landmark in landmarks]) * w)
        min_y = int(min([landmark.y for landmark in landmarks]) * h)
        max_y = int(max([landmark.y for landmark in landmarks]) * h)

        head_extra = int((max_y - min_y) * expand_ratio)
        min_y = max(0, min_y - head_extra)

        min_x, max_x = max(0, min_x), min(w, max_x)
        min_y, max_y = max(0, min_y), min(h, max_y)
        
        min_x, min_y, max_x, max_y = []

        cropped_img = image[min_y:max_y, min_x:max_x]
        return cropped_img if cropped_img.size > 0 else None


