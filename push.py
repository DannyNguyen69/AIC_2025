import os
import json
import numpy as np
from pymongo import MongoClient
import gridfs
from datetime import datetime
import mimetypes
from pathlib import Path

class VideoDataUploader:
    def __init__(self, mongo_uri="mongodb://localhost:27017", db_name="video_processing"):
        """
        Initialize MongoDB connection and GridFS
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.fs = gridfs.GridFS(self.db)
        
    def get_file_type(self, file_path):
        """Determine file type based on extension"""
        ext = Path(file_path).suffix.lower()
        type_mapping = {
            '.json': 'application/json',
            '.npy': 'application/numpy',
            '.jpg': 'image/jpeg', 
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.mp4': 'video/mp4',
            '.txt': 'text/plain',
            '.csv': 'text/csv'
        }
        return type_mapping.get(ext, 'application/octet-stream')
    
    def upload_batch_data(self, batch_root_path):
        """
        Upload entire batch structure to GridFS
        
        Args:
            batch_root_path: Path to data-batch-X folder
        """
        batch_name = os.path.basename(batch_root_path)
        print(f"🚀 Starting upload for {batch_name}")
        
        # Define folder types and their purposes
        folder_types = {
            'clip-features': 'Video clip feature vectors',
            'keyframes': 'Extracted keyframe images', 
            'map-keyframes': 'Keyframe mapping data',
            'metadata': 'Video metadata information',
            'objects': 'Object detection results'
        }
        
        uploaded_count = 0
        
        for folder_name in folder_types:
            folder_path = os.path.join(batch_root_path, folder_name)
            if not os.path.exists(folder_path):
                print(f"⚠️  Folder {folder_name} not found, skipping...")
                continue
                
            print(f"📁 Processing {folder_name}...")
            uploaded_count += self._upload_folder(folder_path, batch_name, folder_name)
        
        print(f"✅ Upload completed! Total files: {uploaded_count}")
        return uploaded_count
    
    def _upload_folder(self, folder_path, batch_name, folder_type):
        """Upload all files in a specific folder"""
        uploaded = 0
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                
                # Create metadata
                metadata = self._create_metadata(file_path, batch_name, folder_type, relative_path)
                
                # Upload file
                if self._upload_single_file(file_path, metadata):
                    uploaded += 1
                    if uploaded % 10 == 0:
                        print(f"   📤 Uploaded {uploaded} files...")
        
        print(f"   ✅ {folder_type}: {uploaded} files uploaded")
        return uploaded
    
    def _create_metadata(self, file_path, batch_name, folder_type, relative_path):
        """Create comprehensive metadata for each file"""
        file_stats = os.stat(file_path)
        path_parts = Path(relative_path).parts
        
        metadata = {
            "batch": batch_name,
            "type": folder_type,
            "original_path": relative_path,
            "file_size": file_stats.st_size,
            "upload_time": datetime.now(),
            "content_type": self.get_file_type(file_path)
        }
        
        # Add specific metadata based on folder type
        if folder_type == "objects" and len(path_parts) >= 2:
            # objects/L01_V001/0001.json
            metadata["video_id"] = path_parts[0]  # L01_V001
            metadata["frame_number"] = Path(path_parts[1]).stem  # 0001
            
        elif folder_type == "keyframes" and len(path_parts) >= 2:
            # keyframes/L01_V001/frame_0001.jpg
            metadata["video_id"] = path_parts[0]
            if "frame" in path_parts[1]:
                metadata["frame_number"] = path_parts[1].replace("frame_", "").split(".")[0]
                
        elif folder_type == "clip-features":
            # clip-features/L01_V001_features.npy
            filename = Path(file_path).stem
            if "_" in filename:
                metadata["video_id"] = filename.split("_")[0]
                
        # Add detection-specific metadata for JSON files
        if file_path.endswith('.json') and folder_type == "objects":
            try:
                with open(file_path, 'r') as f:
                    detection_data = json.load(f)
                    if 'detection_scores' in detection_data:
                        metadata["detection_count"] = len(detection_data['detection_scores'])
                        scores = [float(s) for s in detection_data['detection_scores']]
                        metadata["max_confidence"] = max(scores) if scores else 0
                        metadata["avg_confidence"] = sum(scores) / len(scores) if scores else 0
            except:
                pass
                
        return metadata
    
    def _upload_single_file(self, file_path, metadata):
        """Upload a single file to GridFS"""
        try:
            filename = os.path.basename(file_path)
            
            # Check if file already exists
            existing = self.fs.find_one({
                "filename": filename,
                "metadata.batch": metadata["batch"],
                "metadata.original_path": metadata["original_path"]
            })
            
            if existing:
                print(f"   ⚠️  Skipping {filename} (already exists)")
                return False
            
            with open(file_path, 'rb') as f:
                file_id = self.fs.put(
                    f.read(),
                    filename=filename,
                    metadata=metadata,
                    content_type=metadata["content_type"]
                )
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error uploading {file_path}: {e}")
            return False
    
    def query_examples(self):
        """Show example queries for uploaded data"""
        print("\n🔍 Example queries:")
        
        # Basic queries
        print("\n1. Find all files in a batch:")
        print('   files = fs.find({"metadata.batch": "data-batch-1"})')
        
        print("\n2. Get object detection for specific video:")
        print('   detections = fs.find({"metadata.type": "objects", "metadata.video_id": "L01_V001"})')
        
        print("\n3. Get specific frame detection:")
        print('   frame = fs.find_one({"metadata.video_id": "L01_V001", "metadata.frame_number": "0001"})')
        
        print("\n4. Find high-confidence detections:")
        print('   high_conf = fs.find({"metadata.max_confidence": {"$gt": 0.8}})')
        
        print("\n5. Get all keyframes:")
        print('   keyframes = fs.find({"metadata.type": "keyframes"})')
        
        print("\n6. Download a file:")
        print('   with fs.get(file_id) as grid_out:')
        print('       data = grid_out.read()')
        print('       # For JSON: json.loads(data.decode())')
        print('       # For numpy: np.frombuffer(data)')

    def download_file(self, video_id, frame_number=None, file_type="objects"):
        """
        Download specific file
        
        Args:
            video_id: Video ID (e.g., "L01_V001")
            frame_number: Frame number (e.g., "0001") - optional
            file_type: Type of file ("objects", "keyframes", etc.)
        """
        query = {
            "metadata.type": file_type,
            "metadata.video_id": video_id
        }
        
        if frame_number:
            query["metadata.frame_number"] = frame_number
            
        file_doc = self.fs.find_one(query)
        
        if not file_doc:
            print(f"❌ File not found: {video_id}, frame {frame_number}")
            return None
            
        with self.fs.get(file_doc._id) as grid_out:
            data = grid_out.read()
            
            # Auto-parse based on content type
            if file_doc.content_type == 'application/json':
                return json.loads(data.decode())
            elif file_doc.content_type == 'application/numpy':
                return np.frombuffer(data)
            else:
                return data

    def list_batches(self):
        """List all available batches and their stats"""
        pipeline = [
            {"$group": {
                "_id": "$metadata.batch",
                "total_files": {"$sum": 1},
                "total_size": {"$sum": "$metadata.file_size"},
                "types": {"$addToSet": "$metadata.type"}
            }}
        ]
        
        print("\n📊 Available batches:")
        for batch in self.fs._GridFS__files.aggregate(pipeline):
            size_mb = batch["total_size"] / (1024 * 1024)
            print(f"   {batch['_id']}: {batch['total_files']} files, {size_mb:.1f}MB")
            print(f"      Types: {', '.join(batch['types'])}")

# Usage Example
if __name__ == "__main__":
    # Initialize uploader
    uploader = VideoDataUploader(
        mongo_uri="mongodb://localhost:27017",
        db_name="video_processing"
    )
    
    # Upload batch data
    batch_path = r"D:\Thi_AIC\data-batch-1"  # Thay đổi path này
    uploader.upload_batch_data(batch_path)
    
    # Show available data
    uploader.list_batches()
    
    # Show query examples
    uploader.query_examples()
    
    # Example: Download specific detection data
    # detection_data = uploader.download_file("L01_V001", "0001", "objects")
    # print(detection_data['detection_class_entities'][:5])  # First 5 detected objects