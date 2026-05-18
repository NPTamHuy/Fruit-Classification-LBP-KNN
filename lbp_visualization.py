import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern


# =========================
# IMAGE PATH
# =========================
image_path = "custom_images/pr.jpg"


# =========================
# READ IMAGE
# =========================
image = cv2.imread(image_path)

if image is None:

    print("Khong doc duoc anh!")
    exit()


# =========================
# RGB IMAGE
# =========================
rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# =========================
# GRAYSCALE
# =========================
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# =========================
# RESIZE
# =========================
gray = cv2.resize(gray, (200, 200))

rgb_image = cv2.resize(rgb_image, (200, 200))


# =========================
# LBP PARAMETERS
# =========================
radius = 1
n_points = 8 * radius

# COMPUTE LBP
lbp = local_binary_pattern(
    gray,
    n_points,
    radius,
    method='uniform'
)


# HISTOGRAM
hist, bins = np.histogram(
    lbp.ravel(),
    bins=np.arange(0, n_points + 3),
    range=(0, n_points + 2)
)

# NORMALIZE HISTOGRAM
hist = hist.astype("float")
hist /= (hist.sum() + 1e-6)

# PRINT FEATURE VECTOR
print("\n==============================")
print("LBP FEATURE VECTOR")
print("==============================\n")

print(hist)

print("\nFeature vector length:", len(hist))


# =========================
# CREATE FIGURE
# =========================
fig = plt.figure(figsize=(18, 10))


# =========================
# RGB IMAGE
# =========================
ax1 = fig.add_subplot(2, 3, 1)

img1 = ax1.imshow(rgb_image)

ax1.set_title(
    "Original RGB Image",
    fontsize=14,
    fontweight='bold'
)

ax1.set_xticks([])
ax1.set_yticks([])

for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(3)


# =========================
# GRAYSCALE IMAGE
# =========================
ax2 = fig.add_subplot(2, 3, 2)

img2 = ax2.imshow(gray, cmap='gray')

ax2.set_title(
    "Grayscale Image",
    fontsize=14,
    fontweight='bold'
)

ax2.set_xticks([])
ax2.set_yticks([])

for spine in ax2.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(3)


# =========================
# LBP IMAGE
# =========================
ax3 = fig.add_subplot(2, 3, 3)

img3 = ax3.imshow(lbp, cmap='gray')

ax3.set_title(
    "LBP Texture Pattern",
    fontsize=14,
    fontweight='bold'
)

ax3.set_xticks([])
ax3.set_yticks([])

for spine in ax3.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(3)


# HISTOGRAM
ax4 = fig.add_subplot(2, 1, 2)

bars = ax4.bar(
    bins[:-1],
    hist,
    width=0.5
)

ax4.set_title(
    "LBP Histogram Feature Vector",
    fontsize=14,
    fontweight='bold'
)

ax4.set_xlabel(
    "LBP Pattern",
    fontsize=12
)

ax4.set_ylabel(
    "Frequency",
    fontsize=12
)

text_box = fig.text(
    0.02,
    0.02,
    "",
    fontsize=12,
    bbox=dict(
        facecolor='white',
        edgecolor='black'
    )
)

text_box.set_visible(False)

def on_move(event):

    if event.inaxes not in [ax2, ax3]:
        text_box.set_visible(False)
        fig.canvas.draw_idle()
        return


    # GRAYSCALE INFO
    if event.inaxes == ax2:

        x = int(event.xdata)
        y = int(event.ydata)

        if (
            0 <= x < gray.shape[1]
            and
            0 <= y < gray.shape[0]
        ):

            value = gray[y, x]

            text_box.set_text(
                f"GRAYSCALE\n"
                f"X: {x}  Y: {y}\n"
                f"Intensity: {value}"
            )

            text_box.set_visible(True)

            fig.canvas.draw_idle()


    # LBP INFO
    elif event.inaxes == ax3:

        x = int(event.xdata)
        y = int(event.ydata)

        if (
            0 <= x < lbp.shape[1]
            and
            0 <= y < lbp.shape[0]
        ):

            value = lbp[y, x]

            text_box.set_text(
                f"LBP TEXTURE PATTERN\n"
                f"X: {x}  Y: {y}\n"
                f"LBP Value: {value:.2f}"
            )

            text_box.set_visible(True)

            fig.canvas.draw_idle()


fig.canvas.mpl_connect(
    'motion_notify_event',
    on_move
)
plt.tight_layout()
plt.show()