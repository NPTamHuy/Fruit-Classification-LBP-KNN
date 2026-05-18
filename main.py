import numpy as np

from sklearn.preprocessing import StandardScaler

from src.evaluate import show_predictions

from src.preprocessing import load_images
from src.preprocessing import preprocess_image

from src.feature_extraction import extract_lbp_features
from src.color_features import extract_color_histogram

from src.train_knn import train_knn
from src.train_knn import evaluate_model
from src.train_knn import save_model


# DATASET PATHS
train_path = "dataset/train"
test_path = "dataset/test"


# LOAD DATASET
print("Dang load dataset...")

train_images, train_labels = load_images(train_path)

test_images, test_labels = load_images(test_path)

print("Load dataset thanh cong!")


# EXTRACT TRAIN FEATURES
X_train = []
y_train = []

print("\nDang trich xuat feature train...")

for image, label in zip(train_images, train_labels):

    processed = preprocess_image(image)

    lbp_features = extract_lbp_features(processed)

    color_features = extract_color_histogram(image)

    features = np.hstack([
        lbp_features,
        color_features
    ])

    X_train.append(features)

    y_train.append(label)


# EXTRACT TEST FEATURES
X_test = []
y_test = []

print("\nDang trich xuat feature test...")

for image, label in zip(test_images, test_labels):

    processed = preprocess_image(image)
    lbp_features = extract_lbp_features(processed)
    color_features = extract_color_histogram(image)

    features = np.hstack([
        lbp_features,
        color_features
    ])

    X_test.append(features)
    y_test.append(label)


# CONVERT TO NUMPY ARRAY
X_train = np.array(X_train)
X_test = np.array(X_test)

print("\nSo mau train:", len(X_train))
print("So mau test:", len(X_test))


# FEATURE SCALING
print("\nDang chuan hoa feature...")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# TRAIN MODEL
print("\nDang train KNN...")

model = train_knn(
    X_train,
    y_train,
    k=5
)

# SAVE MODEL
save_model(
    model,
    "models/knn_model.pkl"
)


# EVALUATE MODEL
print("\nDang danh gia model...")

predictions = evaluate_model(
    model,
    X_test,
    y_test
)

# SHOW PREDICTIONS
show_predictions(
    test_images,
    y_test,
    predictions,
    num=5
)