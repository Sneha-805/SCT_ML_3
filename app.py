from flask import Flask,render_template,request
import cv2
import joblib
import numpy as np
import os

app=Flask(__name__)
UPLOAD_FOLDER="static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

model=joblib.load("svm_model.pkl")
joblib.dump(model, "svm_model.pkl", compress=3)
CATEGORIES=["Cat","Dog"]

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
            img = img.reshape(1, -1)                

            result = model.predict(img)
            prediction = CATEGORIES[result]
            filename = file.filename
    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

