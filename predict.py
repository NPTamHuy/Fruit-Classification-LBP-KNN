import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from src.preprocessing import load_images
from src.preprocessing import preprocess_image

from src.feature_extraction import extract_lbp_features
from src.color_features import extract_color_histogram

from src.train_knn import load_model

# LOAD TRAIN DATA
train_path = "dataset/train"

train_images, train_labels = load_images(train_path)

# EXTRACT TRAIN FEATURES
X_train = []

print("Dang tao scaler...")

for image in train_images:

    processed = preprocess_image(image)

    lbp_features = extract_lbp_features(processed)

    color_features = extract_color_histogram(image)

    features = np.hstack([
        lbp_features,
        color_features
    ])

    X_train.append(features)


X_train = np.array(X_train)

# FIT SCALER
scaler = StandardScaler()
scaler.fit(X_train)


# LOAD MODEL
model = load_model("models/knn_model.pkl")


# IMAGE PATH
image_path = r"custom_images/pr.jpg"


# READ IMAGE
image = cv2.imread(image_path)

if image is None:

    print("Khong doc duoc anh!")
    exit()

# DISPLAY IMAGE
display_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# PREPROCESSING
processed = preprocess_image(image)

# FEATURE EXTRACTION
lbp_features = extract_lbp_features(processed)
color_features = extract_color_histogram(image)

feature = np.hstack([
    lbp_features,
    color_features
])


# RESHAPE
feature = np.array(feature).reshape(1, -1)

# FEATURE SCALING
feature = scaler.transform(feature)

# PREDICT
prediction = model.predict(feature)

# RESULT
print("\n======================")
print("KET QUA DU DOAN")
print("======================")

print("Loai trai cay:", prediction[0])

# SHOW IMAGE
plt.figure(figsize=(6, 6))
plt.imshow(display_image)
plt.title(
    f"Prediction: {prediction[0]}"
)
plt.axis("off")
plt.show()