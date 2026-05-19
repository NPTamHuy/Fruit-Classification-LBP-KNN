import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from src.preprocessing import load_images
from src.preprocessing import preprocess_image

from src.feature_extraction import extract_lbp_features
from src.color_features import extract_color_histogram

from src.train_knn import load_model


train_path = "dataset/train"

train_images, train_labels = load_images(train_path)

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

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

model = load_model("models/knn_model.pkl")

image_path = r"custom_images/strawberry1.jpg"

image = cv2.imread(image_path)

if image is None:

    print("Khong doc duoc anh!")
    exit()

display_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

processed = preprocess_image(image)

lbp_features = extract_lbp_features(processed)
color_features = extract_color_histogram(image)

feature = np.hstack([
    lbp_features,
    color_features
])

print("LBP FEATURE VECTOR")
print(lbp_features)
print("\nDo dai LBP vector:", len(lbp_features))

feature = np.array(feature).reshape(1, -1)
feature_scaled = scaler.transform(feature)

prediction = model.predict(feature_scaled)

distances, indices = model.kneighbors(feature_scaled)

print("\n======================")
print("K NEAREST NEIGHBORS")
print("======================")

for i in range(len(indices[0])):

    idx = indices[0][i]

    distance = distances[0][i]

    neighbor_label = train_labels[idx]

    print(f"\nNeighbor {i + 1}")

    print(f"Label    : {neighbor_label}")

    print(f"Distance : {distance:.4f}")

print("\n======================")
print("KET QUA DU DOAN")
print("======================")

print("Loai trai cay:", prediction[0])

plt.figure(figsize=(6, 6))
plt.imshow(display_image)
plt.title(
    f"Prediction: {prediction[0]}"
)
plt.axis("off")
plt.show()