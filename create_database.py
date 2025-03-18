import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# Initialize the face analysis module
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Function to register a new face
def register_face(registration_number, image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    
    if len(faces) == 0:
        return False, "No face detected"
    
    if len(faces) > 1:
        return False, "Multiple faces detected"
    
    # Get the face embedding
    face = faces[0]
    embedding = face.embedding
    
    # Save to database (using simple file storage for demo)
    # In a real system, use a proper database like SQLite, PostgreSQL, etc.
    np.save(f'face_db/{registration_number}.npy', embedding)
    
    # Save metadata (name, ID, etc.)
    with open('face_db/metadata.txt', 'a') as f:
        f.write(f"{registration_number},{os.path.basename(image_path)}\n")
    
    return True, "Face registered successfully"

# Create database directory if it doesn't exist
os.makedirs('face_db', exist_ok=True)

# Example usage
register_face('student001', 'path/to/student_photo.jpg')