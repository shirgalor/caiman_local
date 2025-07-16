import os
import numpy as np
import napari
from imageio.v3 import imread  # תמיכה טובה יותר ב־PNG
from caiman.mmapping import load_memmap
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import matplotlib.pyplot as plt
from skimage.measure import label
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def show_only_selected_rois(label_img, selected_ids):
    mask = np.isin(label_img, selected_ids)
    new_labels = np.where(mask, label_img, 0)
    return new_labels

def launch_napari_viewer(mean_img, correlation_map, A_spatial, C_temporal, Yr_matrix, raw_video):
    viewer = napari.Viewer()

    # 🗾 Add raw video as a time-lapse movie
    viewer.add_image(raw_video, name="Raw Video", colormap="gray", contrast_limits=[np.min(raw_video), np.max(raw_video)])

    # 🗾 Add mean & correlation as separate layers
    viewer.add_image(mean_img, name="Mean Image", colormap="gray", blending="additive")
    viewer.add_image(correlation_map, name="Local Correlation", colormap="gray", blending="additive")

    # 🟦 Build label image for ROIs
    label_img = np.zeros(A_spatial.shape[:2], dtype=np.uint16)
    for i in range(A_spatial.shape[-1]):
        mask = A_spatial[:, :, i] > 0
        label_img[mask & (label_img == 0)] = i + 1

    labels_layer = viewer.add_labels(label_img, name="ROIs (labeled)", opacity=0.5)

    # 🔁 Double-click to show trace
    @labels_layer.mouse_double_click_callbacks.append
    def show_trace(layer, event):
        value = layer.get_value(event.position)
        if value is None or value == 0:
            print("❌ No ROI selected.")
            return

        roi_index = int(value) - 1
        if roi_index >= C_temporal.shape[0]:
            print(f"❌ ROI {roi_index} out of range.")
            return

        trace_c = C_temporal[roi_index]
        mask = A_spatial[:, :, roi_index] > 0
        pix_indices = np.where(mask.flatten(order='F'))[0]
        trace_y = Yr_matrix[pix_indices, :].mean(axis=0)

        plt.figure(figsize=(12, 5))

        # Spatial component A
        plt.subplot(1, 2, 1)
        plt.imshow(A_spatial[:, :, roi_index], cmap='hot')
        plt.title(f"Spatial component A[{roi_index}]")
        plt.colorbar()

        # Temporal traces (C and mean raw Y)
        plt.subplot(1, 2, 2)
        plt.plot(trace_y, label='Raw Y (mean)', color='gray')
        plt.plot(trace_c, label='CNMF C', color='orange')
        plt.title(f"Temporal trace for ROI {roi_index}")
        plt.xlabel("Frame")
        plt.ylabel("Fluorescence")
        plt.legend()

        plt.tight_layout()
        plt.show()

    napari.run()

def tile_images_side_by_side(*images):
    images = [img if img.ndim == 2 else img[..., 0] for img in images]
    max_height = max(img.shape[0] for img in images)
    padded = []
    for img in images:
        h, w = img.shape
        pad_height = max_height - h
        padded_img = np.pad(img, ((0, pad_height), (0, 0)), mode='constant', constant_values=0)
        padded.append(padded_img)
    return np.hstack(padded)

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "cnmf_debug_outputs")

    # Load mean and correlation images
    mean_img = imread(os.path.join(output_dir, "01_mean_image.png"))
    correlation_map = imread(os.path.join(output_dir, "02_local_correlations.png"))

    # Load CNMF results
    cnm = load_CNMF(os.path.join(output_dir, "cnmf_final_output.hdf5"))
    d1, d2 = cnm.dims
    A = cnm.estimates.A.toarray().reshape((d1, d2, -1), order='F')
    C = cnm.estimates.C

    assert A.shape[-1] == C.shape[0], "Mismatch between A and C"

    mmap_filename = "glut_d1_512_d2_512_d3_1_order_C_frames_10.mmap"
    mmap_path = os.path.join(output_dir, mmap_filename)
    Yr, _, _ = load_memmap(mmap_path)

    video_path = os.path.join(os.path.dirname(__file__), '20250409_3_Glut_1mM', '20250409_3_Glut_1mM_TIF_VIDEO.TIF')
    raw_video = imread(video_path)

    launch_napari_viewer(mean_img, correlation_map, A, C, Yr, raw_video)
