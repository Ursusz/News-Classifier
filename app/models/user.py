from app.extensions import db
from sqlalchemy import String, Text, TIMESTAMP
from datetime import datetime, UTC
from flask_login import UserMixin

class USERS(UserMixin, db.Model):
    __tablename__ = 'USERS'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(String(50), unique=True, nullable=False)
    email = db.Column(String(100), unique=True, nullable=False)
    password_hash = db.Column(Text, nullable=False)
    created_at = db.Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(UTC))

    def __repr__(self):
        return f"<Users(id={self.id}, username='{self.username}', email='{self.email}')>"

    def get_id(self):
        return str(self.id)