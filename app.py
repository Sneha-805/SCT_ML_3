from flask import Flask, render_template, request
import cv2
import joblib
import numpy as np
import os
import gdown

MODEL_PATH = "svm_model.pkl"
GOOGLE_DRIVE_FILE_ID = "1iocWFhT4AA7bOcT8LpFe8MuEfSUOmIr0"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    print("Downloading model from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)


app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Try loading the model only if the file exists
model = None
if os.path.exists("svm_model.pkl"):
    model = joblib.load("svm_model.pkl")

CATEGORIES = ["Cat", "Dog"]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100))
            img = img.flatten().reshape(1, -1)

            if model is not None:
                result = model.predict(img)
                prediction = CATEGORIES[result[0]]
            else:
                prediction = "Model not loaded. Please upload the model."

            filename = file.filename

    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
