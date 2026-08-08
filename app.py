import os
import json
import base64
import uuid
from flask import Flask, render_template, request, jsonify
from models import db, User, FaceEmbedding, RecognitionLog
import numpy as np

app = Flask(__name__)

# Configure database dynamically: Use cloud PostgreSQL if DATABASE_URL is set, otherwise fall back to local SQLite
database_url = os.environ.get('DATABASE_URL')
if database_url:
    # SQLAlchemy requires 'postgresql://' but platforms like Render/Heroku often output 'postgres://'
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///faces.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'dev-secret-key-face-rec'

# Initialize Database
db.init_app(app)

# Ensure instance and static directories exist
os.makedirs(app.instance_path, exist_ok=True)
os.makedirs(os.path.join(app.static_folder, 'snapshots'), exist_ok=True)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/users', methods=['GET'])
def get_users():
    users = User.query.order_by(User.name).all()
    return jsonify([user.to_dict() for user in users])

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Delete associated snapshot files if any exist in the recognition logs
    # To keep files clean, we could find logs matching this name and delete their snapshots
    logs = RecognitionLog.query.filter_by(name=user.name).all()
    for log in logs:
        if log.snapshot_path:
            full_path = os.path.join(app.static_folder, log.snapshot_path)
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                except Exception as e:
                    print(f"Error deleting snapshot file: {e}")
                    
    db.session.delete(user)
    # Cascade deletes FaceEmbeddings due to relationship definition
    db.session.commit()
    return jsonify({"status": "success", "message": f"Deleted user {user.name} and their embeddings"})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    if not data or 'name' not in data or 'embedding' not in data:
        return jsonify({"error": "Missing name or embedding"}), 400
    
    name = data['name'].strip()
    embedding = data['embedding']
    
    if not name:
        return jsonify({"error": "Name cannot be empty"}), 400
        
    if len(embedding) != 128:
        return jsonify({"error": "Invalid embedding length. Expected 128 numbers."}), 400
        
    # Check if user exists
    user = User.query.filter_by(name=name).first()
    if not user:
        user = User(name=name)
        db.session.add(user)
        db.session.commit()  # commit to generate ID
        
    # Save the face embedding template
    new_embedding = FaceEmbedding(
        user_id=user.id,
        embedding=json.dumps(embedding)
    )
    db.session.add(new_embedding)
    db.session.commit()
    
    return jsonify({
        "status": "success",
        "message": f"Successfully registered face for {name}",
        "user": user.to_dict()
    })

@app.route('/api/recognize', methods=['POST'])
def recognize():
    data = request.json
    if not data or 'embedding' not in data:
        return jsonify({"error": "Missing embedding"}), 400
    
    input_emb = np.array(data['embedding'], dtype=np.float32)
    if len(input_emb) != 128:
        return jsonify({"error": "Invalid embedding length"}), 400
    
    # Fetch all embeddings
    all_embeddings = FaceEmbedding.query.all()
    if not all_embeddings:
        return jsonify({
            "status": "no_users",
            "name": "Unknown",
            "confidence": 0.0,
            "distance": 99.9
        })
        
    best_match = None
    min_dist = float('inf')
    
    # Match using Euclidean distance
    for db_emb in all_embeddings:
        db_vec = np.array(db_emb.get_vector(), dtype=np.float32)
        dist = np.linalg.norm(db_vec - input_emb)
        if dist < min_dist:
            min_dist = float(dist)
            best_match = db_emb
            
    # Standard threshold: distance < 0.58 is considered the same person
    THRESHOLD = 0.58
    if min_dist < THRESHOLD:
        matched_user = User.query.get(best_match.user_id)
        # Calculate a human-friendly confidence percentage (0 distance = 100%, THRESHOLD distance = 50%)
        confidence = max(0.0, 1.0 - (min_dist / (THRESHOLD * 2)))
        return jsonify({
            "status": "recognized",
            "name": matched_user.name,
            "user_id": matched_user.id,
            "distance": min_dist,
            "confidence": confidence
        })
    else:
        return jsonify({
            "status": "unknown",
            "name": "Unknown",
            "distance": min_dist,
            "confidence": 0.0
        })

@app.route('/api/log_recognition', methods=['POST'])
def log_recognition():
    data = request.json
    if not data or 'name' not in data or 'confidence' not in data:
        return jsonify({"error": "Missing log details"}), 400
        
    name = data['name']
    confidence = float(data['confidence'])
    image_b64 = data.get('image')
    
    snapshot_path = None
    if image_b64:
        try:
            # Handle data:image/jpeg;base64,... header
            if "base64," in image_b64:
                image_data = image_b64.split("base64,")[1]
            else:
                image_data = image_b64
                
            img_bytes = base64.b64decode(image_data)
            
            # Save file in static/snapshots/
            filename = f"snap_{uuid.uuid4().hex[:12]}_{int(db.func.current_timestamp().type.python_type() if hasattr(db.func.current_timestamp(), 'type') else 0)}.jpg"
            # Keep it simple: use timestamp + random tag
            import time
            filename = f"snap_{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
            
            save_dir = os.path.join(app.static_folder, 'snapshots')
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(img_bytes)
                
            snapshot_path = f"snapshots/{filename}"
        except Exception as e:
            print(f"Error saving snapshot image file: {e}")
            
    # Add record to DB
    new_log = RecognitionLog(
        name=name,
        confidence=confidence,
        snapshot_path=snapshot_path
    )
    db.session.add(new_log)
    db.session.commit()
    
    return jsonify({
        "status": "success",
        "log": new_log.to_dict()
    })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    # Return last 30 logs
    logs = RecognitionLog.query.order_by(RecognitionLog.timestamp.desc()).limit(30).all()
    return jsonify([log.to_dict() for log in logs])

@app.route('/api/clear_logs', methods=['POST'])
def clear_logs():
    # Remove files
    logs = RecognitionLog.query.all()
    for log in logs:
        if log.snapshot_path:
            full_path = os.path.join(app.static_folder, log.snapshot_path)
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                except Exception as e:
                    print(f"Error deleting snapshot: {e}")
                    
    # Truncate table
    RecognitionLog.query.delete()
    db.session.commit()
    return jsonify({"status": "success", "message": "Cleared all logs and snapshots"})

if __name__ == '__main__':
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)
