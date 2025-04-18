import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import json
from datetime import datetime
import csv
from pymilvus import connections, Collection, utility

class FaceRecognitionSystem:
    """
    A comprehensive facial recognition system for attendance tracking.
    
    This system utilizes InsightFace for face detection and recognition, and Milvus vector 
    database for efficient facial embedding storage and similarity search. The system can:
    1. Detect faces in real-time from webcam feed
    2. Match detected faces against a database of registered users
    3. Mark attendance for recognized individuals
    4. Provide visual feedback with bounding boxes and recognition confidence
    
    The implementation uses state-of-the-art deep learning models for high accuracy
    facial recognition even under varying lighting conditions and facial expressions.
    """
    
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
        
        # Ensure connection to Milvus
        try:
            connections.connect("default", host="localhost", port="19530")
            print("Connected to Milvus server")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {e}. Please ensure Milvus server is running.")
        
        print("Face recognition system initialized with Milvus vector database.")
    
    def _load_person_database(self):
        """Load person metadata from Milvus"""
        person_db = {}
        
        try:
            if utility.has_collection("face_datastore"):
                collection = Collection("face_datastore")
                collection.load()
                
                results = collection.query(
                    expr="is_embedding == 0",  # 0 instead of False
                    output_fields=["registration_number", "full_name"]
                )
                
                for record in results:
                    reg_num = record["registration_number"]
                    person_db[reg_num] = {
                        "full_name": record["full_name"]
                    }
                
            return person_db
            
        except Exception as e:
            print(f"Error loading metadata from Milvus: {e}")
            return {}
    
    def _find_best_match(self, face_embedding):
        """Find the best matching person using Milvus"""
        try:
            # Use Milvus for vector search
            if not utility.has_collection("face_datastore"):
                print("No face datastore collection found in Milvus")
                return None, None, 0.0
                
            collection = Collection("face_datastore")
            collection.load()
            
            # Search parameters
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            # Search for similar faces
            results = collection.search(
                data=[face_embedding], 
                anns_field="embedding",
                param=search_params,
                limit=1,
                expr="is_embedding == 1",  # 1 instead of True
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
            return None, None, 0.0
    
    def recognize_from_webcam(self, camera_index=0, mark_attendance=True, preview_widget=None, convert_func=None, app_root=None, status_callback=None, stop_flag=None):
        """Run face recognition on webcam feed and optionally mark attendance
        
        Args:
            camera_index: Camera device index
            mark_attendance: Whether to record attendance
            preview_widget: Tkinter widget for displaying camera preview
            convert_func: Function to convert cv2 image to tkinter format
            app_root: Tkinter root window for thread-safe operations
            status_callback: Function to call with status updates
            stop_flag: A callable that returns True when recognition should stop
        """
        # Initialize webcam with specified camera index
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            error_msg = f"Error: Could not open camera {camera_index}."
            print(error_msg)
            if status_callback:
                status_callback(error_msg + "\n")
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
        
        print("Starting face recognition. Use Stop Recognition button to stop.")
        if status_callback:
            status_callback("Starting face recognition. Use Stop Recognition button to stop.\n")
        
        # Variable to track if we should continue running
        running = True
        
        while running:
            # Check if we should stop
            if stop_flag and stop_flag():
                running = False
                if status_callback:
                    status_callback("Recognition stopped by user request.\n")
                break
                
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam.")
                if status_callback:
                    status_callback("Error: Failed to capture image from webcam.\n")
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
                        recognition_msg = f"Marked attendance for {full_name} ({registration_number})\n"
                        print(recognition_msg.strip())
                        if status_callback:
                            status_callback(recognition_msg)
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
            
            # Display the frame - either in Tkinter widget or OpenCV window
            if preview_widget and convert_func and app_root:
                # Get widget dimensions
                if hasattr(preview_widget, 'winfo_width'):
                    width = preview_widget.winfo_width()
                    height = preview_widget.winfo_height()
                    if width > 10 and height > 10:  # Avoid invalid dimensions
                        display_frame = cv2.resize(display_frame, (width, height))
                
                # Convert to Tkinter format and display
                img = convert_func(display_frame)
                
                # Update in a thread-safe manner
                app_root.after(0, lambda: preview_widget.create_image(0, 0, image=img, anchor='nw'))
                app_root.after(0, lambda: setattr(preview_widget, 'image', img))
                
                # Process Tkinter events to keep UI responsive
                try:
                    app_root.update_idletasks()
                    app_root.update()
                except:
                    # If app is being destroyed or there's an error, stop the loop
                    running = False
                    break
            else:
                cv2.imshow("Face Recognition", display_frame)
                # Check for quit key (still keep this for standalone mode)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                
            # Check if preview_widget still exists
            if preview_widget and not preview_widget.winfo_exists():
                running = False
        
        # Release resources
        cap.release()
        if not (preview_widget and convert_func):  # Only destroy windows if using OpenCV display
            cv2.destroyAllWindows()
        
        if mark_attendance and recognized_persons:
            summary = f"\nAttendance summary - {len(recognized_persons)} persons marked present\n"
            summary += f"Attendance saved to {attendance_file}\n"
            print(summary.strip())
            if status_callback:
                status_callback(summary)

# Example usage
if __name__ == "__main__":
    recognition_system = FaceRecognitionSystem(threshold=0.4)
    recognition_system.recognize_from_webcam(mark_attendance=True)