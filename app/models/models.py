from app.extensions import db
from sqlalchemy import Integer, String, ForeignKey
from sqlalchemy.orm import relationship

class DomainName(db.Model):
    __tablename__ = 'DOMAIN_NAME'

    domain_name = db.Column(String(63), nullable=False, primary_key=True)
    real_news = db.Column(db.Integer)
    fake_news = db.Column(db.Integer)

    def __repr__(self):
        return f"<DOMAIN_NAME(ID={self.domain_name})>"

class News(db.Model):
    __tablename__ = 'NEWS'

    news = db.Column(String(2000), nullable=False, primary_key=True)
    domain_name = db.Column(String(63), ForeignKey('DOMAIN_NAME.domain_name'), nullable=False)
    label = db.Column(Integer, nullable=False)
    user = db.Column(String(50), ForeignKey('USERS.username'), nullable=False)

    def __repr__(self):
        return f"<News(id={self.news}')>"
