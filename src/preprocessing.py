import os
import cv2


def load_images(dataset_path):

    images = []
    labels = []

    classes = os.listdir(dataset_path)

    for label in classes:

        class_path = os.path.join(dataset_path, label)

        if not os.path.isdir(class_path):
            continue

        for image_name in os.listdir(class_path):

            image_path = os.path.join(class_path, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            images.append(image)
            labels.append(label)

    return images, labels


def preprocess_image(image, size=(100, 100)):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size)

    return resized