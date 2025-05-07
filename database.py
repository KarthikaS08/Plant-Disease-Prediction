from flask_sqlalchemy import SQLAlchemy

# Initialize the database
db = SQLAlchemy()

# Define the ImageRecord model
class ImageRecord(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

# Function to initialize the database with app context
def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
