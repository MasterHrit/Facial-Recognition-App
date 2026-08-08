from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
import json

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    embeddings = db.relationship('FaceEmbedding', backref='user', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "embedding_count": len(self.embeddings)
        }

class FaceEmbedding(db.Model):
    __tablename__ = 'face_embeddings'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    embedding = db.Column(db.Text, nullable=False)  # JSON serialized array of 128 floats
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def get_vector(self):
        return json.loads(self.embedding)

class RecognitionLog(db.Model):
    __tablename__ = 'recognition_logs'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    snapshot_path = db.Column(db.String(255), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "confidence": round(self.confidence * 100, 1) if self.confidence <= 1.0 else round(self.confidence, 1),
            "timestamp": self.timestamp.isoformat(),
            "snapshot_path": self.snapshot_path
        }
