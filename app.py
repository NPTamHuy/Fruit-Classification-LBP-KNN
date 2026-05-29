"""
=============================================================
  LBP & KNN - VISUALIZATION APP
=============================================================
Cau truc:
    app.py
    src/  preprocessing.py  feature_extraction.py
          color_features.py  train_knn.py  evaluate.py
    dataset/train/   dataset/test/
    models/knn_model.pkl
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections import Counter

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns

from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from src.preprocessing import load_images, preprocess_image
from src.feature_extraction import extract_lbp_features
from src.color_features import extract_color_histogram
from src.train_knn import train_knn, evaluate_model, save_model, load_model


# =============================================================
#  CONFIG
# =============================================================
TRAIN_PATH = "dataset/train"
TEST_PATH  = "dataset/test"
MODEL_PATH = "models/knn_model.pkl"
DEFAULT_K  = 5


# =============================================================
#  TIEN ICH CHUNG
# =============================================================
def extract_features(images):
    X = []
    for img in images:
        proc = preprocess_image(img)
        feat = np.hstack([extract_lbp_features(proc), extract_color_histogram(img)])
        X.append(feat)
    return np.array(X)


def load_and_scale():
    log("Dang load dataset...")
    train_imgs, train_lbls = load_images(TRAIN_PATH)
    test_imgs,  test_lbls  = load_images(TEST_PATH)
    log(f"  Train: {len(train_imgs)}  |  Test: {len(test_imgs)}")
    log("Dang trich xuat feature...")
    X_train = extract_features(train_imgs)
    X_test  = extract_features(test_imgs)
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    log("Xong.")
    return dict(train_imgs=train_imgs, train_lbls=train_lbls,
                test_imgs=test_imgs,   test_lbls=test_lbls,
                X_train=X_train_sc,    X_test=X_test_sc,
                scaler=scaler)


# =============================================================
#  LOG  +  THREAD-SAFE PLOT DISPATCH
# =============================================================
log_widget  = None
_app_ref    = None   # set khi App khoi dong

def log(msg):
    print(msg)
    if log_widget:
        log_widget.configure(state="normal")
        log_widget.insert("end", msg + "\n")
        log_widget.see("end")
        log_widget.configure(state="disabled")


def show_plot(plot_fn):
    """
    Chay plot_fn() tren main thread qua tkinter.after().
    Moi lan goi: dong tat ca figure cu truoc, roi ve figure moi.
    plot_fn phai la ham khong tham so, tra ve list cac figure.
    """
    def _run():
        plt.close("all")
        plot_fn()
        plt.show()
    if _app_ref:
        _app_ref.after(0, _run)


# =============================================================
#  HELPER: tinh uniform LBP
# =============================================================
def uniform_index(binary_str):
    n = len(binary_str)
    transitions = sum(binary_str[i] != binary_str[(i+1) % n] for i in range(n))
    n_ones     = binary_str.count("1")
    is_uniform = transitions <= 2
    label      = n_ones if is_uniform else 9
    return label, transitions, is_uniform, n_ones


# =============================================================
#  PHAN 1 - PIXEL MATRIX
# =============================================================
def run_pixel_matrix(image_path, size=20):
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Loi", "Khong doc duoc anh!"); return
    _g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _g = cv2.resize(_g, (size, size))
    log(f"\n=== PIXEL MATRIX {size}x{size} ===")

    def _plot():
        gray = _g
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                val = gray[i, j]
                # chu mau trang neu nen toi, do neu nen sang
                txt_color = "white" if val < 128 else "red"
                ax.text(j, i, str(val), ha="center", va="center",
                        color=txt_color, fontsize=7, fontweight="bold")
        ax.set_title(f"Grayscale Pixel Matrix ({size}x{size})", fontsize=12, fontweight="bold")
        ax.axis("off")
        # Them colorbar giai thich
        import matplotlib.cm as cm
        sm = plt.cm.ScalarMappable(cmap="gray", norm=plt.Normalize(0, 255))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Gia tri pixel", fontsize=9)
        plt.tight_layout()

    show_plot(_plot)


# =============================================================
#  PHAN 2 - LBP VISUALIZATION
# =============================================================
def run_lbp_visualization(image_path):
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Loi", "Khong doc duoc anh!"); return
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray      = cv2.resize(gray, (200, 200))
    rgb_image = cv2.resize(rgb_image, (200, 200))

    radius = 1; n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, bins = np.histogram(lbp.ravel(),
                              bins=np.arange(0, n_points + 3),
                              range=(0, n_points + 2))
    hist = hist.astype("float"); hist /= (hist.sum() + 1e-6)

    log(f"\n=== LBP VISUALIZATION ===")
    log(f"Feature vector (10 gia tri): {np.round(hist, 4)}")


    def _plot():
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("LBP - Local Binary Pattern Visualization", fontsize=15, fontweight="bold")

        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(rgb_image)
        ax1.set_title("Anh goc (RGB)", fontsize=11, fontweight="bold")
        ax1.set_xticks([]); ax1.set_yticks([])
        for sp in ax1.spines.values(): sp.set_edgecolor("black"); sp.set_linewidth(3)

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(gray, cmap="gray")
        ax2.set_title("Grayscale Image\n(hover: xem gia tri pixel)", fontsize=11, fontweight="bold")
        ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values(): sp.set_edgecolor("black"); sp.set_linewidth(3)

        ax3 = fig.add_subplot(2, 3, 3)
        ax3.imshow(lbp, cmap="gray")
        ax3.set_title("LBP Texture Pattern\n(hover: xem gia tri LBP)", fontsize=11, fontweight="bold")
        ax3.set_xticks([]); ax3.set_yticks([])
        for sp in ax3.spines.values(): sp.set_edgecolor("black"); sp.set_linewidth(3)

        ax4 = fig.add_subplot(2, 1, 2)
        x_ticks    = list(range(len(hist)))
        x_labels   = [f"Bin {i}\n({i} bit 1)" if i < 9 else "Bin 9\n(non-\nuniform)" for i in x_ticks]
        bar_colors = ["#42A5F5"] * 9 + ["#FF7043"]

        bars = ax4.bar(x_ticks, hist, color=bar_colors, edgecolor="white", width=0.65)
        for bar, val in zip(bars, hist):
            if val > 0.01:
                ax4.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.003,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax4.set_xticks(x_ticks); ax4.set_xticklabels(x_labels, fontsize=9)
        ax4.set_title("LBP Uniform Histogram", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Bin index", fontsize=11)
        ax4.set_ylabel("Tan suat (normalized)", fontsize=11)
        ax4.grid(axis="y", linestyle="--", alpha=0.4)
        ax4.set_ylim(0, max(hist) * 1.2)

        # legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#42A5F5", label="Uniform pattern (chuyen bit <= 2)"),
            Patch(facecolor="#FF7043", label="Non-uniform pattern (chuyen bit > 2)"),
        ]
        ax4.legend(handles=legend_elements, loc="upper left", fontsize=9)

        # Hover tooltip
        text_box = fig.text(0.02, 0.02, "", fontsize=10,
                            bbox=dict(facecolor="lightyellow", edgecolor="black"))
        text_box.set_visible(False)

        def on_move(event):
            if event.inaxes == ax2:
                try:
                    x, y = int(event.xdata), int(event.ydata)
                    if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                        text_box.set_text(f"GRAYSCALE\nX:{x}  Y:{y}\nIntensity: {gray[y,x]}")
                        text_box.set_visible(True)
                except: pass
            elif event.inaxes == ax3:
                try:
                    x, y = int(event.xdata), int(event.ydata)
                    if 0 <= x < lbp.shape[1] and 0 <= y < lbp.shape[0]:
                        val = lbp[y, x]
                        idx = int(val) if val <= 8 else 9
                        desc = f"Pattern uniform, so bit 1 = {idx}" if idx < 9 else "Non-uniform pattern"
                        text_box.set_text(f"LBP\nX:{x}  Y:{y}\nUniform index: {idx}")
                        text_box.set_visible(True)
                except: pass
            else:
                text_box.set_visible(False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_move)
        plt.tight_layout()

    show_plot(_plot)


# =============================================================
#  PHAN 3 - LBP NEIGHBORHOOD + UNIFORM
# =============================================================
def run_lbp_neighborhood(image_path, pixel_row=50, pixel_col=50):
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Loi", "Khong doc duoc anh!"); return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 200))
    r = int(np.clip(pixel_row, 1, gray.shape[0] - 2))
    c = int(np.clip(pixel_col, 1, gray.shape[1] - 2))
    center_val = gray[r, c]

    offsets   = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    pos_names = ["tren","tren-phai","phai","duoi-phai","duoi","duoi-trai","trai","tren-trai"]

    neighbor_vals, bits = [], []
    for dr, dc in offsets:
        val = int(gray[r+dr, c+dc])
        neighbor_vals.append(val)
        bits.append(1 if val >= center_val else 0)

    binary_str = "".join(str(b) for b in bits)
    uniform_label, n_transitions, is_uniform, n_ones = uniform_index(binary_str)

    log(f"\n=== LBP NEIGHBORHOOD + UNIFORM ===")
    log(f"Pixel tam    : ({r},{c})  Gia tri: {center_val}")
    log(f"Binary string: {binary_str}")
    log(f"Chuyen bit   : {n_transitions}  |  So bit 1: {n_ones}")
    log(f"Uniform      : {'CO' if is_uniform else 'KHONG'}  |  Histogram bin: {uniform_label}")

    # --- capture vars cho closure ---
    _gray       = gray
    _r, _c      = r, c
    _cv         = center_val
    _offsets    = offsets
    _pos        = pos_names
    _nv         = neighbor_vals
    _bits       = bits
    _bs         = binary_str
    _ul         = uniform_label
    _nt         = n_transitions
    _iu         = is_uniform
    _no         = n_ones

    def _plot():
        # ====================================================
        #  FIGURE 1 - BUOC 1: TINH BIT
        # ====================================================
        fig1 = plt.figure(figsize=(20, 9))
        fig1.suptitle(f"BUOC 1 - Tinh Bit Cho Pixel ({_r},{_c})", fontsize=13, fontweight="bold", y=0.98)
        gs1 = gridspec.GridSpec(2, 5, figure=fig1, hspace=0.55, wspace=0.4)

        # Anh goc
        ax_img = fig1.add_subplot(gs1[:, 0])
        ax_img.imshow(_gray, cmap="gray")
        rect = patches.Rectangle((_c-3.5, _r-3.5), 7, 7,
                                  linewidth=2, edgecolor="red", facecolor="none")
        ax_img.add_patch(rect)
        ax_img.plot(_c, _r, "r+", markersize=12, markeredgewidth=2)
        ax_img.set_title("Anh grayscale", fontsize=10)
        ax_img.axis("off")

        # Zoom 3x3
        ax_zoom = fig1.add_subplot(gs1[0, 1])
        zoom = _gray[_r-1:_r+2, _c-1:_c+2].copy().astype(float)
        ax_zoom.imshow(zoom, cmap="gray", vmin=0, vmax=255)
        pos_map = {(0,0):7,(0,1):0,(0,2):1,(1,0):6,(1,1):"C",(1,2):2,(2,0):5,(2,1):4,(2,2):3}
        for ri in range(3):
            for ci_ in range(3):
                pv  = int(zoom[ri, ci_])
                idx = pos_map[(ri, ci_)]
                if idx == "C":
                    col = "yellow"; lbl = f"C={pv}"
                else:
                    col = "lime" if _bits[idx] == 1 else "red"
                    lbl = f"{pv}\n->{_bits[idx]}"
                ax_zoom.text(ci_, ri, lbl, ha="center", va="center",
                             fontsize=9, fontweight="bold", color=col)
        ax_zoom.set_title(f"Vung 3x3 quanh tam\nC={_cv}", fontsize=10)
        ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
        for sp in ax_zoom.spines.values(): sp.set_edgecolor("black"); sp.set_linewidth(2)

        # Vong tron neighbor
        ax_circle = fig1.add_subplot(gs1[1, 1])
        ax_circle.set_xlim(-2, 2); ax_circle.set_ylim(-2, 2)
        ax_circle.set_aspect("equal"); ax_circle.axis("off")
        ax_circle.set_title("8 Neighbor\n(xanh=1, do=0)", fontsize=10)
        angles = [90, 45, 0, -45, -90, -135, 180, 135]
        for i, ang in enumerate(angles):
            rad = np.deg2rad(ang)
            xp, yp = 1.3*np.cos(rad), 1.3*np.sin(rad)
            col = "#4CAF50" if _bits[i] == 1 else "#F44336"
            ax_circle.add_patch(plt.Circle((xp, yp), 0.32, color=col, zorder=3))
            ax_circle.text(xp, yp, f"{_bits[i]}\n({_nv[i]})",
                           ha="center", va="center", fontsize=7.5, fontweight="bold",
                           color="white", zorder=4)
            ax_circle.plot([0, xp*0.65], [0, yp*0.65], color="gray", lw=0.8, zorder=1)
            ax_circle.text(xp*1.62, yp*1.62, str(i), ha="center", va="center",
                           fontsize=7, color="gray")
        ax_circle.add_patch(plt.Circle((0,0), 0.38, color="#2196F3", zorder=3))
        ax_circle.text(0, 0, f"C={_cv}", ha="center", va="center",
                       fontsize=8, fontweight="bold", color="white", zorder=4)

        # Bang so sanh chi tiet
        ax_table = fig1.add_subplot(gs1[0, 2])
        ax_table.axis("off")
        lines = [f"So sanh neighbor vs tam (C={_cv}):\n",
                 "  #  | Vi tri      | Val | Bit",
                 "  " + "-"*34]
        for i in range(8):
            cmp = ">=" if _bits[i] == 1 else "< "
            lines.append(f"  [{i}] {_pos[i]:<12} {_nv[i]:>3}  {cmp}{_cv} -> {_bits[i]}")
        ax_table.text(0.02, 0.98, "\n".join(lines), transform=ax_table.transAxes,
                      fontsize=9, va="top", fontfamily="monospace",
                      bbox=dict(facecolor="#f5f5f5", edgecolor="gray", boxstyle="round,pad=0.4"))

        # Ket qua binary
        ax_bin = fig1.add_subplot(gs1[1, 2])
        ax_bin.axis("off")
        ax_bin.text(0.5, 0.5,
                    f"Binary = {_bs}",
                    transform=ax_bin.transAxes, fontsize=10, ha="center", va="center",
                    fontfamily="monospace",
                    bbox=dict(facecolor="#E3F2FD", edgecolor="#1565C0", boxstyle="round,pad=0.5"))

        # ====================================================
        #  FIGURE 2 - BUOC 2: UNIFORM CLASSIFICATION
        # ====================================================
        fig2 = plt.figure(figsize=(18, 7))
        fig2.suptitle("BUOC 2 - Phan Loai Uniform LBP", fontsize=14, fontweight="bold")
        gs2 = gridspec.GridSpec(1, 3, figure=fig2, hspace=0.3, wspace=0.45)

        # Panel trai: ve chuyen bit
        ax_trans = fig2.add_subplot(gs2[0, 0])
        ax_trans.axis("off")
        ax_trans.set_title("Dem so lan chuyen bit", fontsize=11, fontweight="bold")

        transitions_pos = [i for i in range(8) if _bs[i] != _bs[(i+1) % 8]]
        cell_w = 0.095; cell_h = 0.18; start_x = 0.04; y_bit = 0.72
        for i, b in enumerate(_bs):
            x    = start_x + i * (cell_w + 0.01)
            is_t = i in transitions_pos
            fc   = "#FFEB3B" if is_t else ("#4CAF50" if b == "1" else "#F44336")
            rp   = patches.FancyBboxPatch(
                (x, y_bit - cell_h/2), cell_w, cell_h,
                boxstyle="round,pad=0.01", facecolor=fc, edgecolor="white", linewidth=1.5,
                transform=ax_trans.transAxes, clip_on=False)
            ax_trans.add_patch(rp)
            ax_trans.text(x + cell_w/2, y_bit, b, ha="center", va="center",
                          fontsize=13, fontweight="bold",
                          color="black" if is_t else "white",
                          transform=ax_trans.transAxes)
            ax_trans.text(x + cell_w/2, y_bit - cell_h/2 - 0.08, f"[{i}]",
                          ha="center", va="top", fontsize=8, color="gray",
                          transform=ax_trans.transAxes)
            if is_t:
                ax_trans.annotate("",
                    xy=(x + cell_w/2, y_bit + cell_h/2 + 0.04),
                    xytext=(x + cell_w/2, y_bit + cell_h/2 + 0.14),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#E65100", lw=2.5))

        trans_pos_str = str(transitions_pos) if transitions_pos else "(khong co)"
        ax_trans.text(0.5, 0.46,
                      f"Vi tri chuyen: {trans_pos_str}\n"
                      f"So lan chuyen: {_nt}",
                      transform=ax_trans.transAxes, ha="center", va="top",
                      fontsize=10, fontfamily="monospace",
                      bbox=dict(facecolor="#FFF3E0", edgecolor="#E65100", boxstyle="round,pad=0.4"))

        # Panel giua: quyet dinh
        ax_dec = fig2.add_subplot(gs2[0, 1])
        ax_dec.axis("off")
        ax_dec.set_title("Quyet dinh Uniform / Non-uniform", fontsize=11, fontweight="bold")
        fc_dec = "#E8F5E9" if _iu else "#FFEBEE"
        bc_dec = "#2E7D32" if _iu else "#C62828"
        verdict = "UNIFORM OK" if _iu else "NON-UNIFORM X"
        rule_text = (
            f"Chuyen bit <= 2  ->  UNIFORM\n"
            f"Chuyen bit  > 2  ->  NON-UNIFORM\n\n"
            f"Binary : {_bs}\n"
            f"Chuyen : {_nt}\n"
            f"Bit 1  : {_no}\n\n"
            f"=> {verdict}"
        )
        ax_dec.text(0.5, 0.5, rule_text, transform=ax_dec.transAxes,
                    ha="center", va="center", fontsize=11, fontfamily="monospace",
                    bbox=dict(facecolor=fc_dec, edgecolor=bc_dec,
                              boxstyle="round,pad=0.6", linewidth=2.5))

        # Panel phai: histogram bin
        ax_hist = fig2.add_subplot(gs2[0, 2])
        ax_hist.set_title("Xep vao histogram bin", fontsize=11, fontweight="bold")
        n_bins       = 10
        bin_labels   = [f"{i}\n({i} bit 1)" for i in range(9)] + ["9\n(non-\nuniform)"]
        bar_colors_h = ["#90CAF9"] * 9 + ["#FFCCBC"]
        highlight_c  = "#1565C0" if _iu else "#BF360C"
        bar_h        = [0.04] * 10
        bar_h[_ul]   = 0.40
        ax_hist.bar(range(n_bins), bar_h, color=bar_colors_h, edgecolor="white", width=0.68)
        ax_hist.bar([_ul], [bar_h[_ul]], color=highlight_c, edgecolor="white", width=0.68)

        ax_hist.set_xticks(range(n_bins))
        ax_hist.set_xticklabels(bin_labels, fontsize=8)
        ax_hist.set_ylabel("Gia tri histogram", fontsize=10)
        ax_hist.set_xlabel("Bin index", fontsize=10)

        offset = 1.5 if _ul < 7 else -2.8
        ax_hist.annotate(
            f"Pixel ({_r},{_c}) vao\nBin [{_ul}]",
            xy=(_ul, bar_h[_ul]),
            xytext=(_ul + offset, 0.30),
            fontsize=10, fontweight="bold", color=highlight_c,
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))


        ax_hist.set_ylim(0, 0.55)
        ax_hist.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()

    show_plot(_plot)


# =============================================================
#  PHAN 4 - PCA 3D
# =============================================================
def run_pca_3d():
    def _compute_and_plot():
        data  = load_and_scale()
        X_all = np.vstack([data["X_train"], data["X_test"]])
        y_all = data["train_lbls"] + data["test_lbls"]
        log("Dang giam chieu PCA 3D...")
        pca = PCA(n_components=3)
        f3d = pca.fit_transform(X_all)
        exp = pca.explained_variance_ratio_ * 100
        log(f"  PC1:{exp[0]:.1f}%  PC2:{exp[1]:.1f}%  PC3:{exp[2]:.1f}%")
        unique_labels = np.unique(y_all)

        def _plot():
            fig = plt.figure(figsize=(12, 10))
            ax  = fig.add_subplot(111, projection="3d")
            ax.set_title(f"3D PCA Feature Space", fontsize=12, fontweight="bold")
            for lbl in unique_labels:
                idx = np.array(y_all) == lbl
                ax.scatter(f3d[idx,0], f3d[idx,1], f3d[idx,2], s=50, alpha=0.7, label=lbl)
            ax.set_xlabel(f"PC1 ({exp[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({exp[1]:.1f}%)")
            ax.set_zlabel(f"PC3 ({exp[2]:.1f}%)")
            ax.legend(loc="upper left", fontsize=8)
            plt.tight_layout()

        show_plot(_plot)

    threading.Thread(target=_compute_and_plot, daemon=True).start()


# =============================================================
#  PHAN 5 - KNN DECISION BOUNDARY
# =============================================================
def run_knn_boundary(k=DEFAULT_K):
    def _compute():
        data = load_and_scale()
        X_train, X_test = data["X_train"], data["X_test"]
        train_lbls, test_lbls = data["train_lbls"], data["test_lbls"]
        unique_labels = sorted(set(train_lbls)); n_classes = len(unique_labels)
        l2i = {l:i for i,l in enumerate(unique_labels)}
        y_tr = np.array([l2i[l] for l in train_lbls])
        y_te = np.array([l2i[l] for l in test_lbls])
        log("Dang PCA 2D va ve decision boundary...")
        pca = PCA(n_components=2)
        Xtr2 = pca.fit_transform(X_train); Xte2 = pca.transform(X_test)
        exp  = pca.explained_variance_ratio_ * 100
        knn2 = KNeighborsClassifier(n_neighbors=k); knn2.fit(Xtr2, y_tr)
        acc  = knn2.score(Xte2, y_te)
        log(f"  Acc 2D: {acc:.4f} (thap hon do PCA mat thong tin)")
        xx, yy = np.meshgrid(
            np.linspace(Xtr2[:,0].min()-1, Xtr2[:,0].max()+1, 300),
            np.linspace(Xtr2[:,1].min()-1, Xtr2[:,1].max()+1, 300))
        Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        bg_c = plt.cm.Set3(np.linspace(0, 1, n_classes))
        pt_c = plt.cm.tab10(np.linspace(0, 1, n_classes))
        cmap = ListedColormap(bg_c[:n_classes])

        def _plot():
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            fig.suptitle(f"KNN Decision Boundary - K={k}  (PCA 2D)", fontsize=13, fontweight="bold")
            for ai, ax in enumerate(axes):
                ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.arange(-0.5, n_classes+0.5, 1))
                ax.contour(xx, yy, Z, colors="white", linewidths=1.0, alpha=0.8,
                           levels=np.arange(-0.5, n_classes+0.5, 1))
                if ai == 0:
                    for i, lbl in enumerate(unique_labels):
                        m = y_tr == i
                        ax.scatter(Xtr2[m,0], Xtr2[m,1], c=[pt_c[i]], edgecolors="white",
                                   s=55, alpha=0.85, label=lbl, zorder=3)
                    ax.set_title("Tap TRAIN", fontsize=11)
                else:
                    preds = knn2.predict(Xte2)
                    n_correct = (preds == y_te).sum()
                    for i, lbl in enumerate(unique_labels):
                        m = y_te == i
                        ok  = m & (preds == y_te)
                        bad = m & (preds != y_te)
                        if ok.any():
                            ax.scatter(Xte2[ok,0], Xte2[ok,1], c=[pt_c[i]], edgecolors="white",
                                       s=70, label=f"{lbl}(dung)", zorder=3)
                        if bad.any():
                            ax.scatter(Xte2[bad,0], Xte2[bad,1], c=[pt_c[i]], edgecolors="red",
                                       linewidths=1.8, s=130, marker="X",
                                       label=f"{lbl}(sai)", zorder=4)
                    ax.set_title(f"Tap TEST - Acc: {acc:.2%}  ({n_correct}/{len(y_te)} dung)", fontsize=11)
                ax.set_xlabel(f"PC1 ({exp[0]:.1f}%)", fontsize=10)
                ax.set_ylabel(f"PC2 ({exp[1]:.1f}%)", fontsize=10)
                ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
                ax.grid(True, linestyle="--", alpha=0.25)
            plt.tight_layout()

        show_plot(_plot)

    threading.Thread(target=_compute, daemon=True).start()


# =============================================================
#  PHAN 6 - K EFFECT  (mac dinh 3 gia tri)
# =============================================================
def run_k_effect(k_list=(1, 5, 9)):
    def _compute():
        data = load_and_scale()
        X_train, X_test = data["X_train"], data["X_test"]
        train_lbls, test_lbls = data["train_lbls"], data["test_lbls"]
        unique_labels = sorted(set(train_lbls)); n_classes = len(unique_labels)
        log(f"\n=== K EFFECT ===")
        results = {}
        for k in k_list:
            knn = KNeighborsClassifier(n_neighbors=k); knn.fit(X_train, train_lbls)
            preds = knn.predict(X_test); acc = accuracy_score(test_lbls, preds)
            cm = confusion_matrix(test_lbls, preds, labels=unique_labels)
            correct = (preds == np.array(test_lbls)).sum()
            results[k] = dict(accuracy=acc, predictions=preds, cm=cm,
                              correct=correct, wrong=len(test_lbls)-correct)
            log(f"  K={k}  Acc={acc:.4f}  Dung={correct}  Sai={len(test_lbls)-correct}")
        acc_vals = [results[k]["accuracy"] for k in k_list]
        best_k   = k_list[int(np.argmax(acc_vals))]
        log(f"  => K tot nhat: {best_k}  Acc: {results[best_k]['accuracy']:.4f}")

        def _plot():
            ncols = len(k_list)

            # Fig 1: accuracy bar + line
            fig1, ax1s = plt.subplots(1, 2, figsize=(13, 5))
            fig1.suptitle("Anh Huong Cua K Den Ket Qua KNN", fontsize=13, fontweight="bold")
            bar_c = ["#4CAF50" if k==best_k else "#EF9A9A" for k in k_list]
            bars  = ax1s[0].bar([f"K={k}" for k in k_list], acc_vals,
                                color=bar_c, edgecolor="white", width=0.5)
            for bar, acc in zip(bars, acc_vals):
                ax1s[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                             f"{acc:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
            ax1s[0].set_ylim(0, 1.15)
            ax1s[0].set_xlabel("Gia tri K", fontsize=11)
            ax1s[0].set_ylabel("Accuracy", fontsize=11)
            ax1s[0].set_title(f"Accuracy theo K  (tot nhat: K={best_k})", fontsize=11)
            ax1s[0].grid(axis="y", linestyle="--", alpha=0.4)
            from matplotlib.patches import Patch


            ax1s[1].plot(list(k_list), acc_vals, marker="o", color="#2196F3", linewidth=2.5,
                         markersize=10, markerfacecolor="white", markeredgewidth=2.5)
            ax1s[1].fill_between(list(k_list), acc_vals, alpha=0.1, color="#2196F3")
            for k, acc in zip(k_list, acc_vals):
                ax1s[1].annotate(
                    f"K={k}\nAcc={acc:.3f}\n({'tot nhat' if k==best_k else ''})",
                    xy=(k, acc), xytext=(0, 18), textcoords="offset points",
                    ha="center", fontsize=9,
                    color="#2E7D32" if k==best_k else "black",
                    fontweight="bold" if k==best_k else "normal",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
            ax1s[1].set_xlabel("K", fontsize=11)
            ax1s[1].set_ylabel("Accuracy", fontsize=11)
            ax1s[1].set_title("Duong Accuracy", fontsize=11)
            ax1s[1].set_xticks(list(k_list))
            ax1s[1].grid(True, linestyle="--", alpha=0.4)
            ax1s[1].set_ylim(max(0, min(acc_vals)-0.08), 1.05)
            plt.tight_layout()
            plt.show()

            # Fig 2: confusion matrix
            fig2, ax2s = plt.subplots(1, ncols, figsize=(4.5*ncols, 5))
            fig2.suptitle("Confusion Matrix Theo Tung K", fontsize=13, fontweight="bold")
            if ncols == 1: ax2s = [ax2s]
            for ax, k in zip(ax2s, k_list):
                sns.heatmap(results[k]["cm"], ax=ax, annot=True, fmt="d", cmap="Blues",
                            xticklabels=unique_labels, yticklabels=unique_labels,
                            linewidths=0.5, linecolor="white", cbar=False)
                star  = " * TOT NHAT" if k==best_k else ""
                color = "#2E7D32" if k==best_k else "black"
                ax.set_title(f"K={k}{star}  Acc:{results[k]['accuracy']:.3f}", fontsize=11, fontweight="bold", color=color)
                ax.set_xlabel("Predicted (du doan)", fontsize=9)
                ax.set_ylabel("True (that te)", fontsize=9)
                ax.tick_params(axis="x", rotation=45); ax.tick_params(axis="y", rotation=0)
            plt.tight_layout()
            plt.show()

            # Fig 3: decision boundary
            l2i = {l:i for i,l in enumerate(unique_labels)}
            y_tr_int = np.array([l2i[l] for l in train_lbls])
            pca = PCA(n_components=2); Xtr2 = pca.fit_transform(X_train)
            exp = pca.explained_variance_ratio_ * 100
            bg_c = plt.cm.Set3(np.linspace(0,1,n_classes))
            pt_c = plt.cm.tab10(np.linspace(0,1,n_classes))
            cmap = ListedColormap(bg_c[:n_classes])
            xx, yy = np.meshgrid(np.linspace(Xtr2[:,0].min()-1,Xtr2[:,0].max()+1,200),
                                 np.linspace(Xtr2[:,1].min()-1,Xtr2[:,1].max()+1,200))
            fig3, ax3s = plt.subplots(1, ncols, figsize=(5*ncols, 5))
            fig3.suptitle("Decision Boundary Theo Tung K (PCA 2D)", fontsize=13, fontweight="bold")
            if ncols == 1: ax3s = [ax3s]
            for ax, k in zip(ax3s, k_list):
                knn2 = KNeighborsClassifier(n_neighbors=k); knn2.fit(Xtr2, y_tr_int)
                Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
                ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.arange(-0.5,n_classes+0.5,1))
                ax.contour(xx, yy, Z, colors="white", linewidths=0.8, alpha=0.6,
                           levels=np.arange(-0.5,n_classes+0.5,1))
                for i, lbl in enumerate(unique_labels):
                    m = y_tr_int == i
                    ax.scatter(Xtr2[m,0], Xtr2[m,1], c=[pt_c[i]], edgecolors="white",
                               linewidths=0.4, s=35, alpha=0.85, label=lbl, zorder=3)
                star  = " *" if k==best_k else ""
                color = "#2E7D32" if k==best_k else "black"
                ax.set_title(f"K={k}{star}", fontsize=11, fontweight="bold", color=color)
                ax.set_xlabel(f"PC1 ({exp[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({exp[1]:.1f}%)")
                ax.legend(fontsize=6, framealpha=0.8); ax.grid(True, linestyle="--", alpha=0.25)
            plt.tight_layout()
            plt.show()

        show_plot(_plot)

    threading.Thread(target=_compute, daemon=True).start()


# =============================================================
#  PHAN 7 - INTERACTIVE PREDICT
# =============================================================
def run_predict(image_path, k=DEFAULT_K):
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Loi", f"Khong tim thay model: {MODEL_PATH}\nHay train truoc!"); return

    def _compute():
        log("\n=== PREDICT ===")
        model = load_model(MODEL_PATH)
        train_imgs, train_lbls = load_images(TRAIN_PATH)
        scaler = StandardScaler(); scaler.fit(extract_features(train_imgs))
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Loi", "Khong doc duoc anh!"); return
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        proc        = preprocess_image(image)
        lbp_feat    = extract_lbp_features(proc)
        color_feat  = extract_color_histogram(image)
        feat        = np.hstack([lbp_feat, color_feat])
        feat_sc     = scaler.transform(feat.reshape(1,-1))
        prediction  = model.predict(feat_sc)[0]
        distances, indices = model.kneighbors(feat_sc, n_neighbors=k)
        neighbor_lbls = [train_lbls[indices[0][i]] for i in range(k)]
        label_count   = Counter(neighbor_lbls)
        log(f"  Ket qua: {prediction}")
        for i in range(k):
            log(f"  Neighbor {i+1}: {train_lbls[indices[0][i]]}  dist={distances[0][i]:.2f}")

        def _plot():
            fig = plt.figure(figsize=(18, 10))
            fig.suptitle(f"Du Doan KNN:  \"{prediction}\"", fontsize=14, fontweight="bold")
            gs = gridspec.GridSpec(2, k+2, figure=fig, hspace=0.5, wspace=0.4)

            ax_in = fig.add_subplot(gs[:,0:2])
            ax_in.imshow(display_img)
            ax_in.set_title(f"Ket qua: \"{prediction}\"", fontsize=12, fontweight="bold", color="#1565C0")
            ax_in.axis("off")
            for sp in ax_in.spines.values(): sp.set_edgecolor("#1565C0"); sp.set_linewidth(3)

            for i in range(k):
                nb_lbl = train_lbls[indices[0][i]]
                nb_rgb = cv2.cvtColor(train_imgs[indices[0][i]], cv2.COLOR_BGR2RGB)
                match  = nb_lbl == prediction
                color  = "#2E7D32" if match else "#C62828"
                ax = fig.add_subplot(gs[0, i+2])
                ax.imshow(nb_rgb)
                ax.set_title(f"Neighbor {i+1}\n{nb_lbl}\ndist={distances[0][i]:.1f}", fontsize=8, color=color, fontweight="bold")
                ax.axis("off")
                for sp in ax.spines.values(): sp.set_edgecolor(color); sp.set_linewidth(2)

            ax_pie = fig.add_subplot(gs[1, 2:4])
            wedge_c = plt.cm.Set2(np.linspace(0,1,len(label_count)))
            _, _, ats = ax_pie.pie(
                list(label_count.values()), labels=list(label_count.keys()),
                autopct="%1.0f%%", startangle=90, colors=wedge_c,
                wedgeprops={"edgecolor":"white","linewidth":1.5})
            for at in ats: at.set_fontsize(10); at.set_fontweight("bold")
            ax_pie.set_title(f"Bo Phieu ({k} Neighbor)", fontsize=11, fontweight="bold")

            ax_lbp = fig.add_subplot(gs[1, 4:])
            bin_lbl = [f"{i}" if i < 9 else "non\nunif" for i in range(len(lbp_feat))]
            ax_lbp.bar(range(len(lbp_feat)), lbp_feat,
                       color=["#42A5F5"]*9+["#FF7043"]*(len(lbp_feat)-9),
                       edgecolor="white", linewidth=0.5)
            ax_lbp.set_xticks(range(len(lbp_feat)))
            ax_lbp.set_xticklabels(bin_lbl, fontsize=8)
            ax_lbp.set_title("LBP Feature Vector", fontsize=10, fontweight="bold")
            ax_lbp.set_xlabel("LBP Bin")
            ax_lbp.set_ylabel("Tan suat (normalized)")
            ax_lbp.grid(axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()

        show_plot(_plot)

    threading.Thread(target=_compute, daemon=True).start()


# =============================================================
#  PHAN 8 - TRAIN
# =============================================================
def run_train(k=DEFAULT_K):
    def _compute():
        data = load_and_scale()
        log(f"\nDang train KNN (K={k})...")
        model = train_knn(data["X_train"], data["train_lbls"], k=k)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        save_model(model, MODEL_PATH)
        log("Dang danh gia...")
        evaluate_model(model, data["X_test"], data["test_lbls"])
    threading.Thread(target=_compute, daemon=True).start()


# =============================================================
#  GUI
# =============================================================
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        global _app_ref
        _app_ref = self
        self.title("LBP & KNN - Visualization App")
        self.resizable(True, True); self.configure(bg="#F5F5F5")
        self.image_path = tk.StringVar(value="")
        self.k_var      = tk.IntVar(value=DEFAULT_K)
        self.pixel_row  = tk.IntVar(value=50)
        self.pixel_col  = tk.IntVar(value=50)
        self.k_list_var = tk.StringVar(value="1,5,9")
        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self, bg="#1565C0", pady=12); header.pack(fill="x")
        tk.Label(header, text="LBP & KNN - Visualization App",
                 font=("Arial",18,"bold"), fg="white", bg="#1565C0").pack()
        tk.Label(header, text="Chon anh, chinh tham so, bam nut de minh hoa",
                 font=("Arial",10), fg="#BBDEFB", bg="#1565C0").pack()

        body  = tk.Frame(self, bg="#F5F5F5"); body.pack(fill="both", expand=True, padx=16, pady=10)
        left  = tk.Frame(body, bg="#F5F5F5"); left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(body, bg="#F5F5F5"); right.pack(side="right", fill="y", padx=(12,0))

        self._section(left, "Chon anh dau vao")
        img_f = tk.Frame(left, bg="#F5F5F5"); img_f.pack(fill="x", pady=(0,4))
        tk.Entry(img_f, textvariable=self.image_path, width=40, font=("Arial",10)).pack(side="left", padx=(0,6))
        tk.Button(img_f, text="Browse...", command=self._browse,
                  font=("Arial",10), bg="#E3F2FD", relief="flat", padx=8, pady=3).pack(side="left")

        self._section(left, "Tham so")
        pf = tk.Frame(left, bg="#F5F5F5"); pf.pack(fill="x", pady=(0,6))
        self._param(pf, "Gia tri K (KNN + boundary):", self.k_var,     0, 1,   30)
        self._param(pf, "Pixel Row (neighborhood):",   self.pixel_row, 2, 1,  198)
        self._param(pf, "Pixel Col (neighborhood):",   self.pixel_col, 4, 1,  198)
        tk.Label(pf, text="Danh sach K (k_effect):", bg="#F5F5F5", font=("Arial",10)).grid(row=6, column=0, sticky="w", pady=3)
        tk.Entry(pf, textvariable=self.k_list_var, width=14, font=("Arial",10)).grid(row=6, column=1, sticky="w", padx=6)
        tk.Label(pf, text="3 gia tri, vi du: 1,5,9", bg="#F5F5F5", font=("Arial",9), fg="gray").grid(row=6, column=2, sticky="w")

        self._section(left, "Cac phan minh hoa")
        buttons = [
            ("1. Pixel Matrix",               self._do_pixel,    "#E3F2FD","#1565C0"),
            ("2. LBP Visualization + Hover",  self._do_lbp_vis,  "#E8F5E9","#2E7D32"),
            ("3. LBP Neighborhood + Uniform", self._do_lbp_nb,   "#FFF3E0","#E65100"),
            ("4. PCA 3D Feature Space",       self._do_pca3d,    "#EDE7F6","#4527A0"),
            ("5. KNN Decision Boundary",      self._do_boundary, "#E0F7FA","#006064"),
            ("6. Anh Huong Cua K (K Effect)", self._do_k_effect, "#FFFDE7","#F57F17"),
            ("7. Du Doan Anh Moi",            self._do_predict,  "#F3E5F5","#6A1B9A"),
        ]
        bf = tk.Frame(left, bg="#F5F5F5"); bf.pack(fill="x")
        for i, (txt, cmd, bg, fg) in enumerate(buttons):
            b = tk.Button(bf, text=txt, command=cmd, font=("Arial",11,"bold"),
                          fg=fg, bg=bg, relief="flat", anchor="w",
                          padx=14, pady=7, cursor="hand2", activebackground=bg)
            b.grid(row=i//2, column=i%2, sticky="ew", padx=4, pady=3)
            bf.columnconfigure(i%2, weight=1)

        self._section(left, "Train / Re-train Model")
        tk.Button(left, text="Train KNN Model", command=self._do_train,
                  font=("Arial",12,"bold"), fg="white", bg="#C62828",
                  relief="flat", padx=16, pady=8, cursor="hand2").pack(fill="x", pady=(0,6))

        self._section(right, "Log")
        global log_widget
        log_widget = tk.Text(right, width=46, font=("Courier",9), state="disabled",
                             bg="#1E1E1E", fg="#DCDCDC", relief="flat", wrap="word")
        log_widget.pack(fill="both", expand=True)
        sb = ttk.Scrollbar(right, command=log_widget.yview); sb.pack(side="right", fill="y")
        log_widget.configure(yscrollcommand=sb.set)
        tk.Button(right, text="Xoa log", command=self._clear_log,
                  font=("Arial",9), relief="flat", bg="#EEE").pack(pady=(4,0))

        log("App san sang!")
        log(f"Train : {TRAIN_PATH}"); log(f"Test  : {TEST_PATH}")
        log(f"Model : {MODEL_PATH}"); log("-"*40)

    def _section(self, p, t):
        tk.Label(p, text=t, font=("Arial",11,"bold"), bg="#F5F5F5", fg="#333").pack(anchor="w", pady=(10,2))
        tk.Frame(p, bg="#BDBDBD", height=1).pack(fill="x", pady=(0,6))

    def _param(self, p, lbl, var, row, col, mx):
        tk.Label(p, text=lbl, bg="#F5F5F5", font=("Arial",10)).grid(row=row, column=0, sticky="w", pady=3)
        tk.Spinbox(p, from_=1, to=mx, textvariable=var, width=8, font=("Arial",10)).grid(row=row, column=1, sticky="w", padx=6)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Chon anh",
            filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp *.webp"),("All files","*.*")])
        if path: self.image_path.set(path); log(f"Da chon: {path}")

    def _clear_log(self):
        log_widget.configure(state="normal"); log_widget.delete("1.0","end")
        log_widget.configure(state="disabled")

    def _img(self):
        path = self.image_path.get().strip()
        if not path: messagebox.showwarning("Chua chon anh","Vui long chon anh truoc!"); return None
        if not os.path.exists(path): messagebox.showerror("Loi",f"Khong tim thay: {path}"); return None
        return path

    def _do_pixel(self):
        p = self._img()
        if p: threading.Thread(target=run_pixel_matrix, args=(p,), daemon=True).start()

    def _do_lbp_vis(self):
        p = self._img()
        if p: threading.Thread(target=run_lbp_visualization, args=(p,), daemon=True).start()

    def _do_lbp_nb(self):
        p = self._img()
        r, c = self.pixel_row.get(), self.pixel_col.get()
        if p: threading.Thread(target=run_lbp_neighborhood, args=(p, r, c), daemon=True).start()

    def _do_color(self):
        pass  # removed

    def _do_pca3d(self):   run_pca_3d()
    def _do_boundary(self): run_knn_boundary(self.k_var.get())
    def _do_k_effect(self):
        try: k_list = tuple(int(x.strip()) for x in self.k_list_var.get().split(","))
        except ValueError: messagebox.showerror("Loi","K list khong hop le. Vi du: 1,5,9"); return
        run_k_effect(k_list)
    def _do_predict(self):
        p = self._img()
        if p: run_predict(p, self.k_var.get())
    def _do_train(self): run_train(self.k_var.get())


# =============================================================
if __name__ == "__main__":
    App().mainloop()