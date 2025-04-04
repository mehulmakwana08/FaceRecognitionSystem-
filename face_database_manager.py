import os
import glob
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

class FaceDatabaseManager:
    def __init__(self, samples_dir='face_samples', db_dir='face_db'):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.samples_dir = samples_dir
        
        # Connect to Milvus
        try:
            connections.connect("default", host="localhost", port="19530")
            print("Connected to Milvus server")
            
            # Create single collection if it doesn't exist
            if not utility.has_collection("face_datastore"):
                self._create_datastore_collection()
                
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {e}. Please ensure Milvus server is running.")
    
    def _create_datastore_collection(self):
        """Create single Milvus collection for all face recognition data"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="registration_number", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="full_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="mobile_number", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="registration_date", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="is_embedding", dtype=DataType.INT8),  # Using INT8 instead of BOOLEAN (0=false, 1=true)
            FieldSchema(name="sample_count", dtype=DataType.INT32),  # Added sample_count field
            FieldSchema(name="sample_index", dtype=DataType.INT32)   # Sample index for embeddings, -1 for metadata
        ]
        schema = CollectionSchema(fields=fields, description="Face recognition datastore")
        
        collection = Collection(name="face_datastore", schema=schema)
        
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("Face datastore collection created successfully")
    
    def _load_metadata(self):
        """Load all person metadata from Milvus"""
        try:
            metadata = {}
            collection = Collection("face_datastore")
            collection.load()
            
            results = collection.query(
                expr="is_embedding == 0",  # 0 instead of False
                output_fields=["registration_number", "full_name", "mobile_number", "registration_date", "sample_count"]
            )
            
            for record in results:
                reg_num = record["registration_number"]
                metadata[reg_num] = {
                    "full_name": record["full_name"],
                    "mobile_number": record["mobile_number"],
                    "registration_date": record["registration_date"],
                    "sample_count": record["sample_count"]
                }
                
            return metadata
            
        except Exception as e:
            print(f"Error loading metadata from Milvus: {e}")
            return {}
    
    def _save_metadata(self, metadata):
        """Save person metadata to Milvus"""
        try:
            collection = Collection("face_datastore")
            
            # Prepare data for insertion
            reg_numbers = []
            full_names = []
            mobile_numbers = []
            reg_dates = []
            sample_counts = []
            is_embeddings = []
            sample_indices = []
            
            for reg_num, data in metadata.items():
                reg_numbers.append(reg_num)
                full_names.append(data.get("full_name", ""))
                mobile_numbers.append(data.get("mobile_number", ""))
                reg_dates.append(data.get("registration_date", self._get_current_date()))
                sample_counts.append(data.get("sample_count", 0))
                is_embeddings.append(False)
                sample_indices.append(-1)
            
            # First check if collection has data
            count = collection.num_entities
            
            if count > 0:
                # Drop and recreate collection instead of deleting individual records
                utility.drop_collection("face_datastore")
                self._create_datastore_collection()
                collection = Collection("face_datastore")
            
            # Insert all records at once
            entities = [
                reg_numbers,
                full_names,
                mobile_numbers,
                reg_dates,
                sample_counts,
                is_embeddings,
                sample_indices
            ]
            
            collection.insert(entities)
            print(f"Successfully saved metadata for {len(reg_numbers)} persons")
            
        except Exception as e:
            print(f"Error saving metadata to Milvus: {e}")
    
    def _get_current_date(self):
        """Get current date in YYYY-MM-DD format"""
        return datetime.now().strftime("%Y-%m-%d")
    
    def register_person(self, registration_number, full_name=None, mobile_number=None):
        """Register a person using multiple face samples"""
        # Ensure directory exists
        os.makedirs(self.samples_dir, exist_ok=True)
        
        # Convert registration number to string
        registration_number = str(registration_number)
        
        # Check if registration number already exists
        collection = Collection("face_datastore")
        collection.load()
        
        results = collection.query(
            expr=f"registration_number == '{registration_number}' && is_embedding == 0",  # 0 instead of false
            output_fields=["id"]
        )
        if len(results) > 0:
            raise ValueError(f"Registration number {registration_number} already exists")
        
        # Find embedding files
        pattern = os.path.join(self.samples_dir, f"{registration_number}_*.npy")
        embedding_files = glob.glob(pattern)
        
        print(f"Looking for samples with pattern: {pattern}")
        print(f"Found {len(embedding_files)} sample files")
        
        if not embedding_files:
            alt_pattern = os.path.join(self.samples_dir, registration_number, "*.npy")
            embedding_files = glob.glob(alt_pattern)
            print(f"Checking alternative pattern: {alt_pattern}")
            print(f"Found {len(embedding_files)} alternative sample files")
            
            if not embedding_files:
                raise ValueError(f"No face samples found for {registration_number}")
        
        # Prepare data for insertion - embeddings
        ids = []
        embeddings = []
        reg_numbers = []
        names = []
        mobile_numbers_list = []
        dates = []
        is_embeddings = []
        sample_counts = []  # Add this list for sample_count field
        sample_indices = []
        
        num_samples = len(embedding_files)
        
        # Add embeddings
        for i, embedding_file in enumerate(embedding_files):
            embedding = np.load(embedding_file)
            entity_id = f"{registration_number}_emb_{i}"
            
            ids.append(entity_id)
            embeddings.append(embedding.tolist())
            reg_numbers.append(registration_number)
            names.append(full_name or "")
            mobile_numbers_list.append(mobile_number or "")
            dates.append(self._get_current_date())
            is_embeddings.append(1)  # 1 instead of True
            sample_counts.append(num_samples)  # Add sample count
            sample_indices.append(i)
        
        # Add metadata record
        ids.append(f"{registration_number}_meta")
        # Add a dummy embedding vector for metadata (required by Milvus)
        dummy_embedding = [0.0] * 512
        embeddings.append(dummy_embedding)
        reg_numbers.append(registration_number)
        names.append(full_name or "")
        mobile_numbers_list.append(mobile_number or "")
        dates.append(self._get_current_date())
        is_embeddings.append(0)  # 0 instead of False
        sample_counts.append(num_samples)  # Add sample count for metadata
        sample_indices.append(-1)
        
        # Structure data for Milvus - now with all 9 fields
        entities = [
            ids,
            embeddings,
            reg_numbers,
            names,
            mobile_numbers_list,
            dates,
            is_embeddings,
            sample_counts,  # Added this field
            sample_indices
        ]
        
        # Insert into Milvus
        collection.insert(entities)
        print(f"Successfully stored {len(embedding_files)} embeddings in single datastore")
        
        # Clean up temporary files
        for file in embedding_files:
            try:
                os.remove(file)
                print(f"Cleaned up temporary file: {file}")
            except:
                pass
                
        return True
    
    def remove_person(self, registration_number):
        """Remove a person from the database"""
        registration_number = str(registration_number)
        
        try:
            collection = Collection("face_datastore")
            collection.load()
            
            # Check if person exists
            results = collection.query(
                expr=f"registration_number == '{registration_number}'",
                output_fields=["id"]
            )
            
            if len(results) == 0:
                raise ValueError(f"Registration number {registration_number} not found")
            
            # Extract primary key values
            ids_to_delete = [record["id"] for record in results]
            
            # Delete using primary key list - Milvus expects this specific format
            if ids_to_delete:
                # For string IDs, we need to wrap each in quotes
                formatted_ids = ",".join([f"'{id_val}'" for id_val in ids_to_delete])
                delete_expr = f"id in [{formatted_ids}]"
                collection.delete(delete_expr)
                
                print(f"Successfully removed person with registration number {registration_number} ({len(ids_to_delete)} records)")
                return True
            else:
                print(f"No records found for registration number {registration_number}")
                return False
                
        except Exception as e:
            print(f"Error removing person: {e}")
            raise
    
    def list_registered_persons(self):
        """List all registered persons from Milvus"""
        try:
            collection = Collection("face_datastore")
            collection.load()
            
            # Debug statement
            print("Querying Milvus for registered persons...")
            
            # Get only metadata records (is_embedding = 0)
            results = collection.query(
                expr="is_embedding == 0",
                output_fields=["registration_number", "full_name", "mobile_number", 
                              "registration_date", "sample_count"]
            )
            
            # Debug statement
            print(f"Found {len(results)} records in database")
            
            metadata = {}
            for record in results:
                reg_num = record["registration_number"]
                metadata[reg_num] = {
                    "full_name": record["full_name"],
                    "mobile_number": record["mobile_number"],
                    "registration_date": record["registration_date"],
                    "sample_count": record["sample_count"]
                }
                # Debug print
                print(f"Added person: {reg_num} - {record['full_name']}")
                
            return metadata
            
        except Exception as e:
            print(f"Error listing persons: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _load_person_database(self):
        """Load person metadata from Milvus"""
        try:
            # Connect to Milvus
            connections.connect("default", host="localhost", port="19530")
            
            # Load metadata from person_metadata collection
            if utility.has_collection("person_metadata"):
                collection = Collection("person_metadata")
                collection.load()
                
                results = collection.query(
                    expr="registration_number != ''",
                    output_fields=["registration_number", "full_name", "registration_date"]
                )
                
                person_db = {}
                for record in results:
                    person_db[record["registration_number"]] = {
                        "name": record["full_name"],
                        "registration_date": record["registration_date"]
                    }
                    
                return person_db
                
        except Exception as e:
            print(f"Error loading metadata from Milvus: {e}")
            
        return {}
        
    def _find_best_match(self, face_embedding):
        """Find best matching person using Milvus vector search"""
        try:
            # Connect to Milvus
            connections.connect("default", host="localhost", port="19530")
            
            if not utility.has_collection("face_embeddings"):
                return None, None, 0
            
            collection = Collection("face_embeddings")
            collection.load()
            
            # Search parameters
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            # Search for the most similar face
            results = collection.search(
                data=[face_embedding],
                anns_field="embedding",
                param=search_params,
                limit=1,
                output_fields=["registration_number", "full_name"]
            )
            
            if results and len(results) > 0 and len(results[0]) > 0:
                match = results[0][0]
                similarity = match.score
                
                if similarity >= self.threshold:
                    return match.entity.get("registration_number"), match.entity.get("full_name"), similarity
            
            return None, None, 0
            
        except Exception as e:
            print(f"Error performing face matching: {e}")
            return None, None, 0

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