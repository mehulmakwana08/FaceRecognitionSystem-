import cv2
import os
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import time
import uuid
import datetime
from pymilvus import utility, FieldSchema, CollectionSchema, DataType, Collection, connections  # Add these import statements

class FaceSampleCollector:
    def __init__(self, save_dir='face_samples', required_samples=5):
        # Initialize face detector and recognition model
        self.app = FaceAnalysis(name='buffalo_l')  # Using InsightFace's buffalo_l model
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.save_dir = save_dir
        self.required_samples = required_samples
        
        # Create directory for storing face samples if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def collect_face_samples(self, registration_number, camera_index=0):
        # Make sure registration_number is always a string
        registration_number = str(registration_number)
        
        # Ensure the save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create person directory
        person_dir = os.path.join(self.save_dir, registration_number)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        # Initialize webcam with specified camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}.")
            return False
        
        # Initialize variables
        collected_samples = 0
        last_capture_time = time.time() - 3  # Wait 3 seconds between captures
        
        print(f"Collecting {self.required_samples} face samples for {registration_number} using camera {camera_index}...")
        print("Please look at the camera and slightly change your head position for each capture.")
        
        while collected_samples < self.required_samples:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam.")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Detect faces
            faces = self.app.get(frame)
            
            # Draw face detection results
            for face in faces:
                bbox = face.bbox.astype(int)
                # Draw rectangle around face
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Draw landmark points
                landmarks = face.landmark_2d_106
                if landmarks is not None:
                    for point in landmarks.astype(int):
                        cv2.circle(display_frame, tuple(point), 1, (0, 0, 255), 2)
            
            # Display the countdown if a face is detected
            current_time = time.time()
            if faces and current_time - last_capture_time >= 2.5:
                time_left = 3 - (current_time - last_capture_time)
                if time_left > 0:
                    cv2.putText(
                        display_frame,
                        f"Capturing in: {int(time_left)+1}",
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )
            
            # Show how many samples have been collected
            cv2.putText(
                display_frame,
                f"Samples: {collected_samples}/{self.required_samples}",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )
            
            # Display the frame
            cv2.imshow("Face Sample Collection", display_frame)
            
            # Capture face when timer is up
            if faces and (current_time - last_capture_time >= 3):
                # Save the primary face (assuming the largest one if multiple detected)
                if len(faces) > 0:
                    # Find the largest face by area
                    largest_area = 0
                    largest_face_idx = 0
                    
                    for i, face in enumerate(faces):
                        bbox = face.bbox.astype(int)
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > largest_area:
                            largest_area = area
                            largest_face_idx = i
                    
                    # Get the largest face and its embedding
                    primary_face = faces[largest_face_idx]
                    
                    # Generate a unique filename
                    filename = f"{registration_number}_{collected_samples}_{uuid.uuid4().hex[:8]}"
                    
                    # Save face image
                    bbox = primary_face.bbox.astype(int)
                    # Add some margin
                    margin_x, margin_y = int((bbox[2] - bbox[0]) * 0.1), int((bbox[3] - bbox[1]) * 0.1)
                    x1 = max(0, bbox[0] - margin_x)
                    y1 = max(0, bbox[1] - margin_y)
                    x2 = min(frame.shape[1], bbox[2] + margin_x)
                    y2 = min(frame.shape[0], bbox[3] + margin_y)
                    
                    face_img = frame[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(person_dir, f"{filename}.jpg"), face_img)
                    
                    # Save embedding
                    np.save(os.path.join(person_dir, f"{filename}.npy"), primary_face.embedding)
                    
                    collected_samples += 1
                    last_capture_time = current_time
                    print(f"Captured sample {collected_samples}/{self.required_samples}")
            
            # Break on ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        if collected_samples == self.required_samples:
            print(f"Successfully collected {collected_samples} samples for {registration_number}")
            return True
        else:
            print(f"Collection interrupted. Collected {collected_samples}/{self.required_samples} samples.")
            return False

    def register_person(self, registration_number, full_name=None, mobile_number=None):
        """Register a person using multiple face samples"""
        # ... existing code ...
        
        # Create collection if it doesn't exist
        if not utility.has_collection("face_embeddings"):
            dim = 512  # InsightFace embedding dimension
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="registration_number", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100)
            ]
            schema = CollectionSchema(fields=fields, description="Face embeddings")
            collection = Collection(name="face_embeddings", schema=schema)
            
            # Create index for vector search
            index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
            collection.create_index(field_name="embedding", index_params=index_params)
        else:
            collection = Collection(name="face_embeddings")
            collection.load()
        
        # Insert embeddings into database
        embeddings = []
        person_dir = os.path.join(self.save_dir, registration_number)
        embedding_files = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith('.npy')]
        for i, embedding_file in enumerate(embedding_files):
            embedding = np.load(embedding_file)
            entity_id = f"{registration_number}_{i}"
            embeddings.append([entity_id, embedding, registration_number, full_name or ""])
        
        collection.insert(embeddings)

    def _load_person_database(self):
        """Load person metadata from Milvus database"""
        person_db = {}
        
        # Connect to Milvus
        try:
            connections.connect("default", host="localhost", port="19530")
            
            # Get metadata from Milvus
            if utility.has_collection("face_embeddings"):
                collection = Collection("face_embeddings")
                collection.load()
                
                # Get all registration numbers
                expr = "registration_number != ''"
                results = collection.query(expr=expr, output_fields=["registration_number", "name"])
                
                # Group by registration number
                for result in results:
                    reg_num = result["registration_number"]
                    name = result["name"]
                    
                    if reg_num not in person_db:
                        person_db[reg_num] = {
                            "name": name,
                            "registration_date": datetime.now().strftime("%Y-%m-%d"),
                        }
            
            return person_db
            
        except Exception as e:
            print(f"Error connecting to Milvus: {e}")
            return {}

    def _find_best_match(self, face_embedding):
        """Find the best matching person using Milvus vector search"""
        try:
            collection = Collection("face_embeddings")
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            results = collection.search(
                data=[face_embedding], 
                anns_field="embedding", 
                param=search_params,
                limit=1,
                output_fields=["registration_number", "name"]
            )
            
            if len(results) > 0 and len(results[0]) > 0:
                match = results[0][0]
                similarity = 1.0 - match.distance / 2  # Convert L2 distance to similarity
                
                if similarity >= self.threshold:
                    return match.entity.get("registration_number"), match.entity.get("name"), similarity
                    
            return None, None, 0.0
                
        except Exception as e:
            print(f"Error during face matching: {e}")
            return None, None, 0.0

# Example usage
if __name__ == "__main__":
    collector = FaceSampleCollector(required_samples=5)
    
    # Get registration number from user
    registration_number = input("Enter the registration number (e.g., student001): ")
    
    # Get camera index (optional)
    try:
        camera_index = int(input("Enter camera index (0, 1, 2, etc.) or press Enter for default: ") or "0")
    except ValueError:
        camera_index = 0
    
    # Collect face samples
    collector.collect_face_samples(registration_number, camera_index)