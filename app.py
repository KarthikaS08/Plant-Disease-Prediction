from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from predictor import predict_plant_disease, predict_tree_species

app = Flask(__name__)

# Set upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'backend/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'JPG'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    theme = request.form.get('theme')

    if not theme:
        return jsonify({"error": "No theme selected"}), 400

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Perform prediction based on theme
            if theme == 'plant':
                prediction = predict_plant_disease(file_path)
            elif theme == 'tree':
                prediction = predict_tree_species(file_path)
            else:
                return jsonify({"error": "Invalid theme selected"}), 400

            return jsonify({
                "filename": filename,
                "prediction": prediction,
                "theme": "Plant Disease" if theme == 'plant' else "Tree Species"
            })

        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)
