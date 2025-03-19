import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import glob
import json
from datetime import datetime
import csv
from pymilvus import connections, Collection, utility

class FaceRecognitionSystem:
    def __init__(self, db_dir='face_db', threshold=0.5):
        """
        Initialize the face recognition system.
        
        Args:
            db_dir: Directory containing the face database
            threshold: Similarity threshold (lower = stricter matching)
        """
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.db_dir = db_dir
        self.threshold = threshold
        
        # Load person database
        self.person_db = self._load_person_database()
        
        print(f"Loaded database with {len(self.person_db)} persons.")
    
    def _load_person_database(self):
        """Load all person embeddings from the database"""
        person_db = {}
        
        # Check if database directory exists
        if not os.path.exists(self.db_dir):
            print(f"Database directory {self.db_dir} not found.")
            return person_db
        
        # Load metadata if available
        metadata_file = os.path.join(self.db_dir, 'metadata.json')
        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Iterate through all person directories
        for registration_number in os.listdir(self.db_dir):
            person_dir = os.path.join(self.db_dir, registration_number)
            
            # Skip metadata file and non-directories
            if not os.path.isdir(person_dir):
                continue
            
            # Get all embedding files for this person
            embedding_files = glob.glob(os.path.join(person_dir, "*.npy"))
            
            if embedding_files:
                # Load all embeddings for this person
                embeddings = [np.load(f) for f in embedding_files]
                
                # Get person name from metadata if available
                name = registration_number
                if "persons" in metadata and registration_number in metadata["persons"]:
                    name = metadata["persons"][registration_number].get("name", registration_number)
                
                person_db[registration_number] = {
                    "name": name,
                    "embeddings": embeddings
                }
        
        return person_db
    
    def _find_best_match(self, face_embedding):
        """Find the best matching person using Milvus"""
        try:
            # Try to use Milvus for vector search
            connections.connect("default", host="localhost", port="19530")
            if utility.has_collection("face_embeddings"):
                collection = Collection("face_embeddings")
                collection.load()
                
                # Search parameters
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
                
                # Search for similar faces
                results = collection.search(
                    data=[face_embedding], 
                    anns_field="embedding",
                    param=search_params,
                    limit=1,
                    output_fields=["registration_number"]
                )
                
                if len(results) > 0 and len(results[0]) > 0:
                    top_match = results[0][0]
                    similarity = top_match.score  # COSINE similarity
                    reg_num = top_match.entity.get("registration_number")
                    
                    # Load metadata to get name
                    metadata = self._load_person_database()
                    name = metadata.get(reg_num, {}).get("full_name", "Unknown")
                    
                    if similarity >= self.threshold:
                        return reg_num, name, similarity
                
                return None, None, 0.0
                
        except Exception as e:
            print(f"Milvus search failed: {e}")
            print("Falling back to file-based search")
        
        # Fall back to original file-based method
        best_match_id = None
        best_match_name = None
        best_similarity = -1
        
        for registration_number, person_data in self.person_db.items():
            # Compare with all embeddings for this person
            similarities = [self._calculate_similarity(face_embedding, ref_emb) 
                           for ref_emb in person_data["embeddings"]]
            
            # Use the highest similarity score
            max_similarity = max(similarities) if similarities else 0
            
            # Update best match if this is better
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match_id = registration_number
                best_match_name = person_data["name"]
        
        # Apply threshold
        if best_similarity < (1 - self.threshold):
            return None, None, best_similarity
        
        return best_match_id, best_match_name, best_similarity
    
    def _calculate_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def recognize_from_webcam(self, camera_index=0, mark_attendance=True):
        """Run face recognition on webcam feed and optionally mark attendance"""
        # Initialize webcam with specified camera index
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}.")
            return
        
        # For attendance tracking
        recognized_persons = set()
        attendance_file = None
        
        if mark_attendance:
            today = datetime.now().strftime("%Y-%m-%d")
            attendance_file = f'attendance_{today}.csv'
            
            # Create/check attendance file
            if not os.path.exists(attendance_file):
                with open(attendance_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Name', 'Time', 'Status'])
        
        print("Starting face recognition. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam.")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Detect faces
            faces = self.app.get(frame)
            
            # Process each detected face
            for face in faces:
                # Get face embedding
                embedding = face.embedding
                
                # Find best match
                registration_number, full_name, similarity = self._find_best_match(embedding)
                
                # Draw bounding box
                bbox = face.bbox.astype(int)
                if registration_number:
                    # Recognized - draw green box
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    
                    # Calculate confidence as percentage
                    confidence = int(similarity * 100)
                    
                    # Display name and confidence
                    text = f"{full_name} ({confidence}%)"
                    cv2.putText(
                        display_frame,
                        text,
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    
                    # Mark attendance if not already marked
                    if mark_attendance and registration_number not in recognized_persons:
                        now = datetime.now().strftime("%H:%M:%S")
                        with open(attendance_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([registration_number, full_name, now, 'Present'])
                        
                        recognized_persons.add(registration_number)
                        print(f"Marked attendance for {full_name} ({registration_number})")
                else:
                    # Unknown - draw red box
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    cv2.putText(
                        display_frame,
                        "Unknown",
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
            
            # Display number of recognized persons
            cv2.putText(
                display_frame,
                f"Recognized: {len(recognized_persons)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            
            # Display the frame
            cv2.imshow("Face Recognition", display_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        if mark_attendance and recognized_persons:
            print(f"\nAttendance summary - {len(recognized_persons)} persons marked present")
            print(f"Attendance saved to {attendance_file}")

# Example usage
if __name__ == "__main__":
    recognition_system = FaceRecognitionSystem(threshold=0.4)
    recognition_system.recognize_from_webcam(mark_attendance=True)