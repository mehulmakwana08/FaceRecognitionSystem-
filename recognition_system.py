import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import glob
import json
from datetime import datetime
import csv

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
        for person_id in os.listdir(self.db_dir):
            person_dir = os.path.join(self.db_dir, person_id)
            
            # Skip metadata file and non-directories
            if not os.path.isdir(person_dir):
                continue
            
            # Get all embedding files for this person
            embedding_files = glob.glob(os.path.join(person_dir, "*.npy"))
            
            if embedding_files:
                # Load all embeddings for this person
                embeddings = [np.load(f) for f in embedding_files]
                
                # Get person name from metadata if available
                name = person_id
                if "persons" in metadata and person_id in metadata["persons"]:
                    name = metadata["persons"][person_id].get("name", person_id)
                
                person_db[person_id] = {
                    "name": name,
                    "embeddings": embeddings
                }
        
        return person_db
    
    def _find_best_match(self, face_embedding):
        """Find the best matching person for a given face embedding"""
        best_match_id = None
        best_match_name = None
        best_similarity = -1
        
        for person_id, person_data in self.person_db.items():
            # Compare with all embeddings for this person
            similarities = [self._calculate_similarity(face_embedding, ref_emb) 
                           for ref_emb in person_data["embeddings"]]
            
            # Use the highest similarity score
            max_similarity = max(similarities) if similarities else 0
            
            # Update best match if this is better
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match_id = person_id
                best_match_name = person_data["name"]
        
        # Apply threshold
        if best_similarity < (1 - self.threshold):
            return None, None, best_similarity
        
        return best_match_id, best_match_name, best_similarity
    
    def _calculate_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def recognize_from_webcam(self, mark_attendance=True):
        """Run face recognition on webcam feed and optionally mark attendance"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
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
                person_id, person_name, similarity = self._find_best_match(embedding)
                
                # Draw bounding box
                bbox = face.bbox.astype(int)
                if person_id:
                    # Recognized - draw green box
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    
                    # Calculate confidence as percentage
                    confidence = int(similarity * 100)
                    
                    # Display name and confidence
                    text = f"{person_name} ({confidence}%)"
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
                    if mark_attendance and person_id not in recognized_persons:
                        now = datetime.now().strftime("%H:%M:%S")
                        with open(attendance_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([person_id, person_name, now, 'Present'])
                        
                        recognized_persons.add(person_id)
                        print(f"Marked attendance for {person_name} ({person_id})")
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