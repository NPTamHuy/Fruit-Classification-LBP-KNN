import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.preprocessing import load_images
from src.preprocessing import preprocess_image

from src.feature_extraction import extract_lbp_features
from src.color_features import extract_color_histogram


# =========================
# DATASET PATH
# =========================

dataset_path = "dataset/train"

# =========================
# LOAD DATASET
# =========================

print("Dang load dataset...")

images, labels = load_images(dataset_path)

print("Load dataset thanh cong!")
print("Tong so anh:", len(images))

# =========================
# EXTRACT FEATURES
# =========================

features = []

print("\nDang trich xuat feature...")

for image in images:

    # Preprocess
    processed = preprocess_image(image)

    # LBP
    lbp_features = extract_lbp_features(processed)

    # Color histogram
    color_features = extract_color_histogram(image)

    # Combine
    combined_features = np.hstack([
        lbp_features,
        color_features
    ])

    features.append(combined_features)

# =========================
# NUMPY ARRAY
# =========================

features = np.array(features)

print("Feature shape:", features.shape)

# =========================
# STANDARDIZE
# =========================

print("\nDang chuan hoa feature...")

scaler = StandardScaler()

features_scaled = scaler.fit_transform(features)

# =========================
# PCA 3D
# =========================

print("Dang giam chieu PCA 3D...")

pca = PCA(n_components=3)

features_3d = pca.fit_transform(features_scaled)

print("PCA shape:", features_3d.shape)

# =========================
# CREATE 3D FIGURE
# =========================

fig = plt.figure(figsize=(12, 10))

ax = fig.add_subplot(111, projection='3d')

# =========================
# UNIQUE LABELS
# =========================

unique_labels = np.unique(labels)

# =========================
# PLOT
# =========================

for label in unique_labels:

    idx = np.array(labels) == label

    ax.scatter(
        features_3d[idx, 0],
        features_3d[idx, 1],
        features_3d[idx, 2],
        s=60,
        alpha=0.8,
        label=label
    )

# =========================
# LABELS
# =========================

ax.set_title("3D Feature Space Visualization")

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

ax.legend()

plt.tight_layout()

plt.show()