import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import datetime
import csv

# Initialize face analysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Load registered face embeddings
def load_face_db():
    face_db = {}
    metadata = {}
    
    # Load metadata
    if os.path.exists('face_db/metadata.txt'):
        with open('face_db/metadata.txt', 'r') as f:
            for line in f:
                person_id, image_name = line.strip().split(',')
                metadata[person_id] = image_name
    
    # Load embeddings
    for file in os.listdir('face_db'):
        if file.endswith('.npy'):
            person_id = os.path.splitext(file)[0]
            embedding = np.load(f'face_db/{file}')
            face_db[person_id] = embedding
    
    return face_db, metadata

# Function to recognize faces and mark attendance
def mark_attendance():
    face_db, metadata = load_face_db()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    # Prepare attendance file
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    attendance_file = f'attendance_{today}.csv'
    
    # Check if file exists, if not create with headers
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Time', 'Status'])
    
    # Keep track of who's already been marked
    marked_attendance = set()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = app.get(frame)
        
        # Process each detected face
        for face in faces:
            # Get embedding of detected face
            embedding = face.embedding
            
            # Compare with database
            min_distance = float('inf')
            recognized_id = None
            
            for person_id, db_embedding in face_db.items():
                # Calculate similarity (cosine similarity)
                similarity = np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
                distance = 1.0 - similarity
                
                if distance < min_distance and distance < 0.4:  # 0.4 is a threshold, adjust as needed
                    min_distance = distance
                    recognized_id = person_id
            
            # Draw rectangle around face
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Put text
            if recognized_id:
                if recognized_id not in marked_attendance:
                    # Mark attendance
                    now = datetime.datetime.now().strftime("%H:%M:%S")
                    with open(attendance_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([recognized_id, now, 'Present'])
                    marked_attendance.add(recognized_id)
                
                cv2.putText(frame, f"ID: {recognized_id}", (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Display
        cv2.imshow('Attendance System', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the attendance system
if __name__ == "__main__":
    mark_attendance()