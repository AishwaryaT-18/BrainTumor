import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Initialize Flask app
app = Flask(__name__)

# Define file paths
path_no_tumor = 'no_tumor'
path_pituitary_tumor = 'pituitary_tumor'
path_meningioma_tumor = 'meningioma_tumor'
path_glioma_tumor = 'glioma_tumor'

tumor_check = {'no_tumor': 0, 'pituitary_tumor': 1, 'meningioma_tumor': 2, 'glioma_tumor': 3}

# Load data function
def load_data():
    x = []
    y = []

    for cls in tumor_check:
        if cls == 'no_tumor':
            path = path_no_tumor
        elif cls == 'pituitary_tumor':
            path = path_pituitary_tumor
        elif cls == 'meningioma_tumor':
            path = path_meningioma_tumor
        elif cls == 'glioma_tumor':
            path = path_glioma_tumor

        for j in os.listdir(path):
            image = cv2.imread(os.path.join(path, j), 0)
            if image is None:
                print(f"Failed to load image: {os.path.join(path, j)}")
                continue
            image = cv2.resize(image, (200, 200))
            x.append(image)
            y.append(tumor_check[cls])

    x = np.array(x)
    y = np.array(y)
    return x, y

# Prepare data
x, y = load_data()
x_update = x.reshape(len(x), -1)
x_train, x_test, y_train, y_test = train_test_split(x_update, y, random_state=10, test_size=0.3)
x_train = x_train / 255
x_test = x_test / 255

pca = PCA(.98)
pca_train = pca.fit_transform(x_train)
pca_test = pca.transform(x_test)

logistic = LogisticRegression(C=0.1)
logistic.fit(pca_train, y_train)

sv = SVC()
sv.fit(pca_train, y_train)

@app.route("/")
def hello_world():
    return render_template('home.html')

# Execute function
@app.route("/execute_python_function", methods=["POST"])
def execute_python_function():
    try:
        # Get the uploaded file
        file = request.files['image']
        img_bytes = file.read()

        # Read the image
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (200, 200))
        img_processed = img_resized.reshape(1, -1) / 255

        # Make prediction
        predicted_label = sv.predict(pca.transform(img_processed))
        tumor_type = {0: 'no_tumor', 1: 'pituitary_tumor', 2: 'meningioma_tumor', 3: 'glioma_tumor'}[predicted_label[0]]

        return tumor_type
    except Exception as e:
        return f"Error processing image: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001)
