import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

class RobustPersonReidentification:
    def __init__(self):
        """
        Initialize robust person re-identification system
        Focuses on handling variations and reducing false negatives
        """
        # Face recognition
        self.face_threshold = 0.6  # Relaxed threshold
        
        # Body detection
        self.person_detector = YOLO('yolov8n.pt')
        
        # Deep feature extractor
        self.feature_extractor = self._load_feature_extractor()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_feature_extractor(self):
        """
        Load deep feature extraction model
        """
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the last classification layer
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()
        return feature_extractor
    
    def extract_face_features(self, image):
        """
        Multi-approach face feature extraction
        More robust to variations
        """
        # Ensure image is RGB
        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Multiple face detection approaches
        # 1. Standard face recognition
        face_locations = face_recognition.face_locations(
            rgb_image, 
            model='hog'  # Faster, more robust
        )
        
        # 2. Compute face encodings
        face_encodings = face_recognition.face_encodings(
            rgb_image, 
            face_locations
        )
        
        return face_encodings
    
    def extract_body_features(self, image):
        """
        Comprehensive body feature extraction
        """
        # Detect persons
        results = self.person_detector(image)
        
        body_features = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = image[y1:y2, x1:x2]
                
                # Color histogram features
                hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
                h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
                s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
                
                # Normalize histograms
                cv2.normalize(h_hist, h_hist)
                cv2.normalize(s_hist, s_hist)
                
                # Deep feature extraction
                input_tensor = self.transform(person_crop).unsqueeze(0)
                with torch.no_grad():
                    deep_features = self.feature_extractor(input_tensor)
                    deep_features = deep_features.squeeze().numpy()
                
                body_features.append({
                    'color_hist': np.concatenate([h_hist.flatten(), s_hist.flatten()]),
                    'deep_features': deep_features
                })
        
        return body_features
    
    def create_person_signature(self, initial_image):
        """
        Create a comprehensive person signature
        """
        # Face features
        face_features = self.extract_face_features(initial_image)
        
        # Body features
        body_features = self.extract_body_features(initial_image)
        
        return {
            'face_features': face_features,
            'body_features': body_features
        }
    
    def match_person(self, current_image, person_signature, threshold=0.5):
        """
        Advanced person matching with multiple feature comparisons
        Designed to reduce false negatives
        """
        # Extract current image features
        current_face_features = self.extract_face_features(current_image)
        current_body_features = self.extract_body_features(current_image)
        
        # Face matching with relaxed criteria
        face_match_scores = []
        for stored_face in person_signature['face_features']:
            for current_face in current_face_features:
                face_distance = face_recognition.face_distance([stored_face], current_face)
                face_match_scores.append(1 - face_distance[0])
        
        # Body feature matching
        body_match_scores = []
        if person_signature['body_features'] and current_body_features:
            for stored_body in person_signature['body_features']:
                for current_body in current_body_features:
                    # Color histogram comparison
                    color_hist_similarity = cv2.compareHist(
                        stored_body['color_hist'], 
                        current_body['color_hist'], 
                        cv2.HISTCMP_CORREL
                    )
                    
                    # Deep feature cosine similarity
                    deep_feature_similarity = np.dot(
                        stored_body['deep_features'], 
                        current_body['deep_features']
                    ) / (np.linalg.norm(stored_body['deep_features']) * 
                         np.linalg.norm(current_body['deep_features']))
                    
                    # Combined body matching score
                    body_match = 0.6 * color_hist_similarity + 0.4 * deep_feature_similarity
                    body_match_scores.append(body_match)
        
        # Compute overall match scores
        face_match = max(face_match_scores) if face_match_scores else 0
        body_match = max(body_match_scores) if body_match_scores else 0
        
        # Weighted combination with lower threshold
        combined_score = 0.6 * face_match + 0.4 * body_match
        
        return combined_score > threshold, combined_score

# Example usage
def main():
    import time
    
    # Initialize re-identification system
    start_time = time.time()
    reid_system = RobustPersonReidentification()
    print(f"System initialization time: {time.time() - start_time:.4f} seconds")
    
    # Load initial image
    initial_image = cv2.imread('/home/zainey/reid/WhatsApp Image 2025-03-10 at 13.26.16 (3).jpeg')
    
    # Create person signature
    start_time = time.time()
    person_signature = reid_system.create_person_signature(initial_image)
    print(f"Signature creation time: {time.time() - start_time:.4f} seconds")
    
    # Load current image
    current_image = cv2.imread('/home/zainey/reid/WhatsApp Image 2025-03-10 at 13.26.16 (4).jpeg')
    
    # Match person
    start_time = time.time()
    is_matched, confidence = reid_system.match_person(
        current_image, 
        person_signature
    )
    print(f"Matching time: {time.time() - start_time:.4f} seconds")
    
    print(f"Person matched: {is_matched}")
    print(f"Confidence: {confidence}")

if __name__ == "__main__":
    main()

# Dependencies:
"""
Required libraries:
pip install opencv-python
pip install face_recognition
pip install ultralytics
pip install torch torchvision
"""