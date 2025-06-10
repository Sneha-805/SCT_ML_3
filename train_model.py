import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from sklearn.svm import SVC
import joblib

DATADIR="dataset"
CATEGORIES=["Dog","Cat"]

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)

    print(f"Reading from folder: {path}")
    count = 0
    

    for img in os.listdir(path):
        if not img.endswith(".jpg"):
            continue

        img_path = os.path.join(path, img)
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img_array is None:
            print(f"Failed to read: {img_path}")
            continue

        resized = cv2.resize(img_array, (100, 100)).flatten()
        data.append(resized)
        labels.append(class_num)
        count += 1

        if count >= 1000:
            break

print(f"Successfully loaded {len(data)} images")

  

X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,random_state=42)
svm=SVC(kernel="linear")
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
print(classification_report(y_test,y_pred))

os.makedirs("model",exist_ok=True)
joblib.dump(svm,"svm_model.pkl")