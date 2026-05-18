import matplotlib.pyplot as plt
import cv2


def show_predictions(images, labels, predictions, num=5):

    plt.figure(figsize=(15, 5))

    for i in range(num):

        plt.subplot(1, num, i + 1)

        # Convert BGR -> RGB
        rgb_image = cv2.cvtColor(
            images[i],
            cv2.COLOR_BGR2RGB
        )

        plt.imshow(rgb_image)

        plt.title(
            f"True: {labels[i]}\nPred: {predictions[i]}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()