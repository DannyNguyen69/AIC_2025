# app.py - Flask Web Application for Video Frame Search
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import json
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from pymongo import MongoClient
import gridfs
from datetime import datetime
from collections import defaultdict, Counter
import cv2
import io

class VideoSearchWebApp:
    def __init__(self, mongo_uri="mongodb://localhost:27017", db_name="video_processing"):
        """Initialize Flask app and database connection"""
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Database connection
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.fs = gridfs.GridFS(self.db)
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup all web routes"""
        
        @self.app.route('/')
        def index():
            """Main search page"""
            return render_template('index.html')
        
        @self.app.route('/api/search', methods=['POST'])
        def api_search():
            """API endpoint for search"""
            try:
                data = request.get_json()
                search_term = data.get('query', '').strip()
                confidence = float(data.get('confidence', 0.7))
                max_results = int(data.get('max_results', 50))
                video_filter = data.get('video_filter', '')
                
                if not search_term:
                    return jsonify({'error': 'Search term is required'}), 400
                
                # Perform search
                results = self.search_frames(
                    search_term, confidence, max_results, video_filter
                )
                
                # Convert images to base64 for web display
                web_results = []
                for result in results['results'][:max_results]:
                    # Get frame image
                    image_b64 = self.get_frame_image_b64(
                        result['video_id'], 
                        result['frame_number']
                    )
                    
                    web_result = {
                        'video_id': result['video_id'],
                        'frame_number': result['frame_number'],
                        'confidence': result['confidence'],
                        'entity': result['entity'],
                        'class_name': result['class_name'],
                        'bbox': result['bbox'],
                        'image': image_b64
                    }
                    web_results.append(web_result)
                
                return jsonify({
                    'success': True,
                    'total_matches': results['total_matches'],
                    'results': web_results,
                    'search_term': search_term,
                    'confidence_threshold': confidence
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stats')
        def api_stats():
            """Get dataset statistics"""
            try:
                stats = self.get_dataset_stats()
                return jsonify(stats)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/suggestions')
        def api_suggestions():
            """Get search suggestions"""
            try:
                suggestions = self.get_popular_objects()
                return jsonify({'suggestions': suggestions})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/frame/<video_id>/<frame_number>')
        def api_frame_detail(video_id, frame_number):
            """Get detailed frame information"""
            try:
                frame_data = self.get_frame_details(video_id, frame_number)
                return jsonify(frame_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def search_frames(self, search_term, confidence_threshold=0.7, max_results=50, video_filter=''):
        """Search frames by object name"""
        query = {"metadata.type": "objects"}
        if video_filter:
            query["metadata.video_id"] = {"$regex": video_filter, "$options": "i"}
        
        results = []
        processed_count = 0
        
        for file_doc in self.fs.find(query):
            try:
                with self.fs.get(file_doc._id) as grid_out:
                    data = json.loads(grid_out.read().decode())
                
                entities = data.get("detection_class_entities", [])
                class_names = data.get("detection_class_names", [])
                scores = [float(s) for s in data.get("detection_scores", [])]
                boxes = data.get("detection_boxes", [])
                
                # Search for matching objects
                for i, (entity, class_name, score) in enumerate(zip(entities, class_names, scores)):
                    if score >= confidence_threshold:
                        if (search_term.lower() in entity.lower() or 
                            search_term.lower() in class_name.lower()):
                            
                            result = {
                                "video_id": file_doc.metadata.get("video_id"),
                                "frame_number": file_doc.metadata.get("frame_number"),
                                "entity": entity,
                                "class_name": class_name,
                                "confidence": score,
                                "bbox": [float(x) for x in boxes[i]] if i < len(boxes) else [],
                                "filename": file_doc.filename
                            }
                            results.append(result)
                            
                            if len(results) >= max_results:
                                break
                
                processed_count += 1
                if len(results) >= max_results:
                    break
                    
            except Exception as e:
                continue
        
        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "total_matches": len(results),
            "results": results,
            "processed_files": processed_count
        }
    
    def get_frame_image_b64(self, video_id, frame_number):
        """Get frame image as base64 string"""
        frame_str = str(frame_number).zfill(4)
        
        query = {
            "metadata.type": "keyframes",
            "metadata.video_id": video_id,
            "metadata.frame_number": frame_str
        }
        
        file_doc = self.fs.find_one(query)
        
        if not file_doc:
            # Try alternative patterns
            alt_query = {
                "metadata.type": "keyframes",
                "metadata.video_id": video_id
            }
            
            for alt_file in self.fs.find(alt_query):
                if frame_str in alt_file.filename:
                    file_doc = alt_file
                    break
        
        if not file_doc:
            return None
        
        # Get image data and convert to base64
        with self.fs.get(file_doc._id) as grid_out:
            image_data = grid_out.read()
        
        return base64.b64encode(image_data).decode()
    
    def get_frame_details(self, video_id, frame_number):
        """Get all details for a specific frame"""
        frame_str = str(frame_number).zfill(4)
        
        # Get object detection data
        obj_query = {
            "metadata.type": "objects",
            "metadata.video_id": video_id,
            "metadata.frame_number": frame_str
        }
        
        obj_file = self.fs.find_one(obj_query)
        objects_data = []
        
        if obj_file:
            try:
                with self.fs.get(obj_file._id) as grid_out:
                    data = json.loads(grid_out.read().decode())
                
                entities = data.get("detection_class_entities", [])
                class_names = data.get("detection_class_names", [])
                scores = [float(s) for s in data.get("detection_scores", [])]
                boxes = data.get("detection_boxes", [])
                
                for i, (entity, class_name, score) in enumerate(zip(entities, class_names, scores)):
                    objects_data.append({
                        "entity": entity,
                        "class_name": class_name,
                        "confidence": score,
                        "bbox": [float(x) for x in boxes[i]] if i < len(boxes) else []
                    })
                    
            except Exception as e:
                pass
        
        # Get frame image
        image_b64 = self.get_frame_image_b64(video_id, frame_number)
        
        return {
            "video_id": video_id,
            "frame_number": frame_str,
            "objects": objects_data,
            "image": image_b64,
            "total_objects": len(objects_data)
        }
    
    def get_dataset_stats(self):
        """Get overall dataset statistics"""
        # Count videos
        videos = self.db.fs.files.distinct("metadata.video_id", {"metadata.type": "objects"})
        
        # Count frames
        frames = self.db.fs.files.count_documents({"metadata.type": "objects"})
        
        # Count keyframes
        keyframes = self.db.fs.files.count_documents({"metadata.type": "keyframes"})
        
        # Get most common objects (sample)
        object_counter = Counter()
        sample_files = list(self.fs.find({"metadata.type": "objects"}).limit(1000))
        
        for file_doc in sample_files:
            try:
                with self.fs.get(file_doc._id) as grid_out:
                    data = json.loads(grid_out.read().decode())
                
                entities = data.get("detection_class_entities", [])
                scores = [float(s) for s in data.get("detection_scores", [])]
                
                for entity, score in zip(entities, scores):
                    if score >= 0.5:
                        object_counter[entity] += 1
                        
            except Exception:
                continue
        
        return {
            "total_videos": len(videos),
            "total_frames": frames,
            "total_keyframes": keyframes,
            "top_objects": object_counter.most_common(20),
            "sample_videos": videos[:10]
        }
    
    def get_popular_objects(self):
        """Get popular object names for search suggestions"""
        # Get most common objects from a sample
        object_counter = Counter()
        sample_files = list(self.fs.find({"metadata.type": "objects"}).limit(500))
        
        for file_doc in sample_files:
            try:
                with self.fs.get(file_doc._id) as grid_out:
                    data = json.loads(grid_out.read().decode())
                
                entities = data.get("detection_class_entities", [])
                scores = [float(s) for s in data.get("detection_scores", [])]
                
                for entity, score in zip(entities, scores):
                    if score >= 0.6:
                        object_counter[entity] += 1
                        
            except Exception:
                continue
        
        # Return top 50 most common objects
        return [obj for obj, count in object_counter.most_common(50)]
    
    def run(self, host='127.0.0.1', port=5000, debug=True):
        """Run the Flask application"""
        print(f"🚀 Starting Video Search Web App on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Create templates directory and files
def create_templates():
    """Create HTML templates for the web app"""
    
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Main HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Frame Search</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        .search-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem 0;
            color: white;
        }
        
        .frame-card {
            transition: transform 0.2s;
            cursor: pointer;
        }
        
        .frame-card:hover {
            transform: scale(1.05);
        }
        
        .confidence-badge {
            position: absolute;
            top: 10px;
            right: 10px;
        }
        
        .loading {
            display: none;
        }
        
        .search-stats {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .suggestions {
            margin-top: 1rem;
        }
        
        .suggestion-tag {
            display: inline-block;
            background: #e9ecef;
            padding: 0.25rem 0.5rem;
            margin: 0.25rem;
            border-radius: 15px;
            cursor: pointer;
            font-size: 0.8rem;
        }
        
        .suggestion-tag:hover {
            background: #6c757d;
            color: white;
        }
    </style>
</head>
<body>
    <div class="search-container">
        <div class="container">
            <div class="row">
                <div class="col-12 text-center">
                    <h1><i class="fas fa-search"></i> Video Frame Search</h1>
                    <p class="lead">Search through your video dataset to find specific objects and scenes</p>
                </div>
            </div>
            
            <div class="row justify-content-center">
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-body">
                            <form id="searchForm">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label for="searchQuery" class="form-label">Search Term</label>
                                            <input type="text" class="form-control" id="searchQuery" 
                                                   placeholder="e.g., car, person, traffic sign..." required>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="mb-3">
                                            <label for="confidence" class="form-label">Min Confidence</label>
                                            <select class="form-control" id="confidence">
                                                <option value="0.5">0.5</option>
                                                <option value="0.6">0.6</option>
                                                <option value="0.7" selected>0.7</option>
                                                <option value="0.8">0.8</option>
                                                <option value="0.9">0.9</option>
                                            </select>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="mb-3">
                                            <label for="maxResults" class="form-label">Max Results</label>
                                            <select class="form-control" id="maxResults">
                                                <option value="20">20</option>
                                                <option value="50" selected>50</option>
                                                <option value="100">100</option>
                                                <option value="200">200</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label for="videoFilter" class="form-label">Video Filter (optional)</label>
                                            <input type="text" class="form-control" id="videoFilter" 
                                                   placeholder="e.g., L01_V001">
                                        </div>
                                    </div>
                                    <div class="col-md-6 d-flex align-items-end">
                                        <button type="submit" class="btn btn-primary btn-lg w-100">
                                            <i class="fas fa-search"></i> Search
                                        </button>
                                    </div>
                                </div>
                            </form>
                            
                            <!-- Search Suggestions -->
                            <div class="suggestions">
                                <small class="text-muted">Popular searches:</small>
                                <div id="suggestionTags"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Loading indicator -->
    <div class="container">
        <div class="loading text-center" id="loadingIndicator">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>Searching through frames...</p>
        </div>
    </div>
    
    <!-- Search Results -->
    <div class="container">
        <div id="searchResults"></div>
        <div id="resultsContainer" class="row"></div>
    </div>
    
    <!-- Frame Detail Modal -->
    <div class="modal fade" id="frameModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Frame Details</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="frameModalBody">
                    <!-- Frame details will be loaded here -->
                </div>
            </div>
        </div>
    </div>
    
    <!-- Dataset Stats -->
    <div class="container mt-4">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-chart-bar"></i> Dataset Statistics</h5>
            </div>
            <div class="card-body" id="datasetStats">
                Loading statistics...
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
    <script>
        // Global variables
        let currentResults = [];
        
        // Load initial data
        document.addEventListener('DOMContentLoaded', function() {
            loadDatasetStats();
            loadSuggestions();
        });
        
        // Search form submission
        document.getElementById('searchForm').addEventListener('submit', function(e) {
            e.preventDefault();
            performSearch();
        });
        
        // Perform search
        function performSearch() {
            const query = document.getElementById('searchQuery').value.trim();
            const confidence = parseFloat(document.getElementById('confidence').value);
            const maxResults = parseInt(document.getElementById('maxResults').value);
            const videoFilter = document.getElementById('videoFilter').value.trim();
            
            if (!query) {
                alert('Please enter a search term');
                return;
            }
            
            // Show loading
            document.getElementById('loadingIndicator').style.display = 'block';
            document.getElementById('searchResults').innerHTML = '';
            document.getElementById('resultsContainer').innerHTML = '';
            
            // Make API call
            fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    confidence: confidence,
                    max_results: maxResults,
                    video_filter: videoFilter
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loadingIndicator').style.display = 'none';
                
                if (data.success) {
                    displayResults(data);
                } else {
                    alert('Search error: ' + data.error);
                }
            })
            .catch(error => {
                document.getElementById('loadingIndicator').style.display = 'none';
                alert('Network error: ' + error);
            });
        }
        
        // Display search results
        function displayResults(data) {
            currentResults = data.results;
            
            // Show stats
            const statsHtml = `
                <div class="search-stats">
                    <div class="row">
                        <div class="col-md-3">
                            <strong>Search Term:</strong> ${data.search_term}
                        </div>
                        <div class="col-md-3">
                            <strong>Total Matches:</strong> ${data.total_matches}
                        </div>
                        <div class="col-md-3">
                            <strong>Confidence:</strong> ≥ ${data.confidence_threshold}
                        </div>
                        <div class="col-md-3">
                            <strong>Showing:</strong> ${data.results.length} results
                        </div>
                    </div>
                </div>
            `;
            document.getElementById('searchResults').innerHTML = statsHtml;
            
            // Show frame results
            const resultsContainer = document.getElementById('resultsContainer');
            resultsContainer.innerHTML = '';
            
            data.results.forEach((result, index) => {
                const frameCard = createFrameCard(result, index);
                resultsContainer.appendChild(frameCard);
            });
        }
        
        // Create frame card
        function createFrameCard(result, index) {
            const col = document.createElement('div');
            col.className = 'col-md-3 col-sm-6 mb-4';
            
            const confidenceClass = result.confidence >= 0.8 ? 'bg-success' : 
                                  result.confidence >= 0.6 ? 'bg-warning' : 'bg-secondary';
            
            col.innerHTML = `
                <div class="card frame-card" onclick="showFrameDetails('${result.video_id}', '${result.frame_number}')">
                    <div class="position-relative">
                        <img src="data:image/jpeg;base64,${result.image}" 
                             class="card-img-top" style="height: 200px; object-fit: cover;">
                        <span class="badge ${confidenceClass} confidence-badge">
                            ${(result.confidence * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="card-body">
                        <h6 class="card-title">${result.video_id}</h6>
                        <p class="card-text">
                            <small>Frame: ${result.frame_number}</small><br>
                            <strong>${result.entity}</strong><br>
                            <small class="text-muted">${result.class_name}</small>
                        </p>
                    </div>
                </div>
            `;
            
            return col;
        }
        
        // Show frame details in modal
        function showFrameDetails(videoId, frameNumber) {
            fetch(`/api/frame/${videoId}/${frameNumber}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error loading frame details: ' + data.error);
                    return;
                }
                
                let objectsHtml = '';
                data.objects.forEach(obj => {
                    const confidenceClass = obj.confidence >= 0.8 ? 'success' : 
                                          obj.confidence >= 0.6 ? 'warning' : 'secondary';
                    
                    objectsHtml += `
                        <div class="mb-2">
                            <span class="badge bg-${confidenceClass}">${(obj.confidence * 100).toFixed(1)}%</span>
                            <strong>${obj.entity}</strong>
                            <small class="text-muted">(${obj.class_name})</small>
                        </div>
                    `;
                });
                
                const modalBody = document.getElementById('frameModalBody');
                modalBody.innerHTML = `
                    <div class="row">
                        <div class="col-md-8">
                            <img src="data:image/jpeg;base64,${data.image}" 
                                 class="img-fluid" style="max-height: 400px;">
                        </div>
                        <div class="col-md-4">
                            <h6>Frame Information</h6>
                            <p><strong>Video:</strong> ${data.video_id}</p>
                            <p><strong>Frame:</strong> ${data.frame_number}</p>
                            <p><strong>Objects:</strong> ${data.total_objects}</p>
                            
                            <h6>Detected Objects</h6>
                            ${objectsHtml}
                        </div>
                    </div>
                `;
                
                // Show modal
                const modal = new bootstrap.Modal(document.getElementById('frameModal'));
                modal.show();
            })
            .catch(error => {
                alert('Network error: ' + error);
            });
        }
        
        // Load dataset statistics
        function loadDatasetStats() {
            fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('datasetStats').innerHTML = 'Error loading stats: ' + data.error;
                    return;
                }
                
                let topObjectsHtml = '';
                data.top_objects.slice(0, 10).forEach(([obj, count]) => {
                    topObjectsHtml += `<span class="badge bg-secondary me-1">${obj} (${count})</span>`;
                });
                
                const statsHtml = `
                    <div class="row">
                        <div class="col-md-3">
                            <h6><i class="fas fa-video"></i> Videos</h6>
                            <h4>${data.total_videos}</h4>
                        </div>
                        <div class="col-md-3">
                            <h6><i class="fas fa-images"></i> Frames</h6>
                            <h4>${data.total_frames.toLocaleString()}</h4>
                        </div>
                        <div class="col-md-3">
                            <h6><i class="fas fa-key"></i> Keyframes</h6>
                            <h4>${data.total_keyframes.toLocaleString()}</h4>
                        </div>
                        <div class="col-md-3">
                            <h6><i class="fas fa-percentage"></i> Coverage</h6>
                            <h4>${((data.total_keyframes/data.total_frames)*100).toFixed(1)}%</h4>
                        </div>
                    </div>
                    <hr>
                    <h6>Most Common Objects</h6>
                    <div>${topObjectsHtml}</div>
                `;
                
                document.getElementById('datasetStats').innerHTML = statsHtml;
            })
            .catch(error => {
                document.getElementById('datasetStats').innerHTML = 'Error loading statistics';
            });
        }
        
        // Load search suggestions
        function loadSuggestions() {
            fetch('/api/suggestions')
            .then(response => response.json())
            .then(data => {
                if (data.suggestions) {
                    const container = document.getElementById('suggestionTags');
                    container.innerHTML = '';
                    
                    data.suggestions.slice(0, 20).forEach(suggestion => {
                        const tag = document.createElement('span');
                        tag.className = 'suggestion-tag';
                        tag.textContent = suggestion;
                        tag.onclick = () => {
                            document.getElementById('searchQuery').value = suggestion;
                        };
                        container.appendChild(tag);
                    });
                }
            })
            .catch(error => {
                console.log('Could not load suggestions:', error);
            });
        }
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("✅ HTML template created at templates/index.html")

# Main execution
if __name__ == "__main__":
    # Create templates
    create_templates()
    
    # Initialize and run the web app
    app = VideoSearchWebApp(
        mongo_uri="mongodb://localhost:27017",
        db_name="video_processing"
    )
    
    print("🌐 Video Frame Search Web Application")
    print("=" * 50)
    print("Features:")
    print("✅ Real-time frame search")
    print("✅ Confidence filtering")
    print("✅ Visual results with thumbnails")
    print("✅ Frame detail modal")
    print("✅ Dataset statistics")
    print("✅ Search suggestions")
    print("✅ Responsive design")
    
    # Run the app
    app.run(host='127.0.0.1', port=5000, debug=True)