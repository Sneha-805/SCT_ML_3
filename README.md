🐱🐶 Cat vs. Dog Image Classification with SVM
This project is part of Task 3 from my internship at SkillCraft Technology. The goal is to build an SVM (Support Vector Machine) model that classifies images as either a cat or a dog, and deploy the model using Flask.

📌 Features
📥 Upload grayscale image of a cat or dog

🤖 SVM model trained on resized 100x100 images

📈 Simple Flask-based UI for real-time prediction

📦 Model trained with ~17,500 images (cats and dogs from Kaggle dataset)

🌐 Deployable on Render

🛠️ Tech Stack
Python, OpenCV

Scikit-learn (SVM)

Flask

HTML/CSS (Jinja2)

Git & GitHub

📁 Dataset
Dataset used: Kaggle Cats vs Dogs Dataset

~8861 cat images

~8754 dog images

Preprocessed and flattened to grayscale 100x100 images

🚀 How to Run Locally
Clone the repository:

bash
Copy
Edit
git clone https://github.com/Sneha-805/cat-dog-svm-flask.git
cd cat-dog-svm-flask
Install the requirements:

bash
Copy
Edit
pip install -r requirements.txt
Make sure svm_model.pkl is present (or run train_model.py to generate it).

Start the app:

bash
Copy
Edit
python app.py
Open in your browser:

cpp
Copy
Edit
http://127.0.0.1:5000
🧪 Sample Predictions
✅ Correct predictions on most clean dog and cat images

❌ Misclassifications may occur with noisy backgrounds or uncommon angles

📂 Folder Structure
csharp
Copy
Edit
cat-dog-svm-flask/
│
├── static/
│   └── uploads/        ← Uploaded image storage
│
├── templates/
│   └── index.html      ← Web page for image upload
│
├── dataset/            ← Dataset folder with 'train' inside
│   └── train/
│       ├── cat.0.jpg ...
│       └── dog.0.jpg ...
│
├── app.py              ← Flask app
├── train_model.py      ← Training script
├── svm_model.pkl       ← Trained model
├── README.md           ← You’re reading it!
└── requirements.txt    ← Python dependencies
👩‍💻 Author
Sneha Mudda
B.Tech CSE, IIIT RK Valley
LinkedIn | GitHub
