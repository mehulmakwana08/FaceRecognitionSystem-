import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import glob
import json
import shutil
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

class FaceDatabaseManager:
    def __init__(self, samples_dir='face_samples', db_dir='face_db'):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.samples_dir = samples_dir
        self.db_dir = db_dir
        
        connections.connect("default", host="localhost", port="19530")
        # Create directories if they don't exist
        for directory in [samples_dir, db_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Load existing database metadata
        self.metadata_file = os.path.join(db_dir, 'metadata.json')
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from JSON file or create if it doesn't exist"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {"persons": {}}
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_person(self, registration_number, full_name=None, mobile_number=None):
        """Register a person using multiple face samples"""
        # Check if sample directory exists for this person
        person_sample_dir = os.path.join(self.samples_dir, registration_number)
        if not os.path.exists(person_sample_dir):
            print(f"No samples found for {registration_number}. Please collect samples first.")
            return False
        
        # Get all embedding files for this person
        embedding_files = glob.glob(os.path.join(person_sample_dir, "*.npy"))
        if not embedding_files:
            print(f"No embeddings found for {registration_number}.")
            return False
        
        # Create folder for this person in the database if it doesn't exist
        person_db_dir = os.path.join(self.db_dir, registration_number)
        if not os.path.exists(person_db_dir):
            os.makedirs(person_db_dir)
        
        # Copy all embeddings and images to database
        for emb_file in embedding_files:
            # Get corresponding image file
            base_name = os.path.splitext(os.path.basename(emb_file))[0]
            img_file = os.path.join(person_sample_dir, f"{base_name}.jpg")
            
            # Copy files if image exists
            if os.path.exists(img_file):
                shutil.copy2(emb_file, os.path.join(person_db_dir, os.path.basename(emb_file)))
                shutil.copy2(img_file, os.path.join(person_db_dir, os.path.basename(img_file)))
        
        # Update metadata
        self.metadata["persons"][registration_number] = {
            "name": full_name if full_name else registration_number,
            "mobile": mobile_number if mobile_number else "",
            "sample_count": len(embedding_files),
            "registration_date": self._get_current_date()
        }
        self._save_metadata()
        
        print(f"Successfully registered {registration_number} with {len(embedding_files)} face samples.")
        return True
    
    def remove_person(self, registration_number):
        """Remove a person from the database"""
        person_db_dir = os.path.join(self.db_dir, registration_number)
        
        if os.path.exists(person_db_dir):
            shutil.rmtree(person_db_dir)
            
            # Update metadata
            if registration_number in self.metadata["persons"]:
                del self.metadata["persons"][registration_number]
                self._save_metadata()
            
            print(f"Successfully removed {registration_number} from the database.")
            return True
        else:
            print(f"{registration_number} not found in the database.")
            return False
    
    def list_registered_persons(self):
        """List all registered persons with their info"""
        if not self.metadata["persons"]:
            print("No persons registered in the database.")
            return []
        
        print("\nRegistered Persons:")
        print("-" * 80)
        print(f"{'Registration #':<15} {'Full Name':<20} {'Mobile':<15} {'Samples':<10} {'Registration Date':<20}")
        print("-" * 80)
        
        for registration_number, info in self.metadata["persons"].items():
            print(f"{registration_number:<15} {info['name']:<20} {info.get('mobile', ''):<15} {info['sample_count']:<10} {info['registration_date']:<20}")
        
        return list(self.metadata["persons"].keys())
    
    def _get_current_date(self):
        """Get current date in YYYY-MM-DD format"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")

# Example usage
if __name__ == "__main__":
    db_manager = FaceDatabaseManager()
    
    # Sample operations
    while True:
        print("\nFace Database Manager")
        print("1. Register person from samples")
        print("2. List registered persons")
        print("3. Remove person")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            registration_number = input("Enter registration number: ")
            full_name = input("Enter full name (optional): ")
            mobile_number = input("Enter mobile number (optional): ")
            db_manager.register_person(registration_number, full_name, mobile_number)
        elif choice == "2":
            db_manager.list_registered_persons()
        elif choice == "3":
            registration_number = input("Enter registration number to remove: ")
            confirm = input(f"Are you sure you want to remove {registration_number}? (y/n): ")
            if confirm.lower() == 'y':
                db_manager.remove_person(registration_number)
        elif choice == "4":
            break
        else:
            print("Invalid choice. Please try again.")