import os
import numpy as np
import tempfile
from tifffile import imread
import matplotlib.pyplot as plt
from caiman.source_extraction.cnmf import cnmf, params
from caiman.mmapping import load_memmap
from caiman.summary_images import local_correlations
from skimage.measure import label, regionprops
from scipy.ndimage import center_of_mass


# ---------- Utility: Save image or plot ----------
def save_image(arr, name, output_dir, cmap='gray', show_live=False):
    plt.figure()
    if arr.ndim == 2:
        im = plt.imshow(arr, cmap=cmap)
        plt.colorbar(im)
    elif arr.ndim == 1:
        plt.plot(arr)
        plt.xlabel("Time (frames)")
        plt.ylabel("Fluorescence")
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    plt.title(name)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{name}.png")
    plt.savefig(save_path)

    if show_live:
        plt.pause(0.1)
    plt.close()
    print(f"🖼 Saved: {save_path}")

def show_patch_on_mean_image(mean_img, patch_coords, roi_masks=None, title='Current Patch'):
    x0, y0, width, height = patch_coords
    plt.figure()
    plt.imshow(mean_img, cmap='gray')
    rect = plt.Rectangle((x0, y0), width, height, edgecolor='red', facecolor='none', linewidth=2)
    plt.gca().add_patch(rect)

    if roi_masks:
        for mask in roi_masks:
            if not np.any(mask):
                continue
            cy, cx = center_of_mass(mask)
            plt.plot(cx, cy, 'go', markersize=4)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.pause(0.1)
    plt.close()

# ---------- Wrapper for CNMF Debug Mode ----------
class CNMFWrapper:
    def __init__(self, cnmf_obj, mmap_path, dims, output_dir):
        self.cnmf = cnmf_obj
        self.mmap_path = mmap_path
        self.dims = dims
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_overlay(self, Cn, A, output_path, show_live=False):
        plt.figure(figsize=(8, 8))
        plt.imshow(Cn, cmap='gray')
        plt.title("ROIs on Correlation Map")

        A_binary = (A > 0).astype(int)
        A_sum = np.sum(A_binary, axis=2)
        labeled = label(A_sum)

        for region in regionprops(labeled):
            y, x = region.centroid
            plt.plot(x, y, 'ro', markersize=2)

        plt.tight_layout()
        plt.axis('off')
        plt.xlim(0, Cn.shape[1])
        plt.ylim(Cn.shape[0], 0)
        plt.savefig(output_path)
        if show_live:
            plt.pause(0.1)
        plt.close()
        print(f"🖼 Saved ROI overlay: {output_path}")

    def run(self):
        Yr, dims, T = load_memmap(self.mmap_path)

        Y_avg = np.mean(Yr, axis=1).reshape(self.dims, order='F')
        save_image(Y_avg, "01_mean_image", self.output_dir, show_live=False)

        Cn = local_correlations(Yr.T.reshape((T, *dims)), swap_dim=False)
        Cn[np.isnan(Cn)] = 0
        save_image(Cn, "02_local_correlations", self.output_dir, show_live=False)

        Yr = self.cnmf.preprocess(Yr)
        save_image(self.cnmf.estimates.sn.reshape(self.dims), "03_noise_std_estimate", self.output_dir)

        self.cnmf.dims = self.dims
        def debug_show_patch_with_rois(coords, roi_masks, title=''):
            show_patch_on_mean_image(Y_avg, coords, roi_masks=roi_masks, title=title)
        self.cnmf._debug_show_patch = debug_show_patch_with_rois

        self.cnmf.fit_file(self.mmap_path)


        # >>> כאן תכניס את הקוד הבא:
        A_initial = self.cnmf.estimates.Ab.toarray().reshape((*self.dims, -1), order='F')  # לפני המיזוג
        A_final = self.cnmf.estimates.A.toarray().reshape((*self.dims, -1), order='F')     # אחרי המיזוג

        # שמור תווית לכל שכבה
        label_img_initial = np.zeros(self.dims, dtype=np.uint16)
        label_img_final = np.zeros(self.dims, dtype=np.uint16)

        for i in range(A_initial.shape[-1]):
            mask = A_initial[:, :, i] > 0
            label_img_initial[mask & (label_img_initial == 0)] = i + 1

        for i in range(A_final.shape[-1]):
            mask = A_final[:, :, i] > 0
            label_img_final[mask & (label_img_final == 0)] = i + 1

        # שמור כתמונות PNG – תוכל גם להעלות לנפארי אח"כ
        save_image(label_img_initial, "06_labels_before_merge", self.output_dir, cmap='viridis')
        save_image(label_img_final,   "07_labels_after_merge",  self.output_dir, cmap='viridis')

        A_final = self.cnmf.estimates.A.toarray().reshape((*self.dims, -1), order='F')
        C_final = self.cnmf.estimates.C

        for i in range(A_final.shape[-1]):
            save_image(A_final[:, :, i], f"07_A_after_merge_{i}", self.output_dir)
            if i < C_final.shape[0]:
                save_image(C_final[i], f"08_temporal_trace_component_{i}", self.output_dir, cmap='viridis')

        Yr_full, _, _ = load_memmap(self.mmap_path)
        for i in range(A_final.shape[-1]):
            mask = A_final[:, :, i] > 0
            pix_indices = np.where(mask.flatten(order='F'))[0]
            Y_roi = Yr_full[pix_indices, :]  
            mean_trace = Y_roi.mean(axis=0)
            save_image(mean_trace, f"10_Y_mean_trace_{i}", self.output_dir)
        b_spatial = self.cnmf.estimates.b.reshape((*self.dims, -1), order='F')
        for j in range(b_spatial.shape[-1]):
            save_image(b_spatial[:, :, j], f"11_background_b_{j}", self.output_dir)

        f_temporal = self.cnmf.estimates.f
        for j in range(f_temporal.shape[0]):
            save_image(f_temporal[j], f"12_background_f_{j}", self.output_dir, cmap='viridis')

        final_path = os.path.join(self.output_dir, "cnmf_final_output.hdf5")
        if hasattr(self.cnmf, "_debug_show_patch"):
            del self.cnmf._debug_show_patch

        self.cnmf.save(final_path)
        print(f"✅ Debug run complete. Output saved to {final_path}")
        self.save_overlay(Cn, A_final, os.path.join(self.output_dir, "09_overlay_rois_on_Cn.png"), show_live=True)

# ---------- MAIN ----------
if __name__ == "__main__":
    import os
    import numpy as np
    from tifffile import imread
    from caiman.source_extraction.cnmf import cnmf, params

    file_path = os.path.join(os.path.dirname(__file__), '20250409_3_Glut_1mM', '20250409_3_Glut_1mM_TIF_VIDEO.TIF')
    Y = imread(file_path) 
    Y = Y[:10]
    T, d1, d2 = Y.shape
    dims = (d1, d2)

    output_dir = os.path.join(os.path.dirname(__file__), "cnmf_debug_outputs")
    os.makedirs(output_dir, exist_ok=True)

    mmap_base = f"glut_d1_{d1}_d2_{d2}_d3_1_order_C_frames_{T}"
    mmap_path = os.path.join(output_dir, mmap_base + '.mmap')
    npy_path = os.path.join(output_dir, mmap_base + '.npy')

   
    np.save(npy_path, Y.astype(np.float32))
    fp = np.memmap(mmap_path, mode='w+', dtype=np.float32, shape=Y.size)
    fp[:] = Y.flatten(order='C')[:]
    fp.flush()  # ✅ force it to write to disk
    del fp  # ✅ close the memmap

    opts = params.CNMFParams()
    opts.set('data', {
        'dims': dims,
        'fr': 30,
        'p': 1,
        'is3D': False,
        'data_format': 'mmap',
        'fnames': [mmap_path]
    })
    opts.set('init', {
        'gSig': [5, 5],
        'nb': 2,
        'method_init': 'greedy_roi',
        'K': 60
    })
    opts.set('patch', {
        'rf': 40,
        'stride': 20,
        'only_init': True
    })
    opts.set('spatial', {'merge_thresh': 0.7})
    opts.set('temporal', {'p': 1})
    opts.set('online', {'only_init': False})

    # יצירת אובייקט CNMF
    cnm = cnmf.CNMF(n_processes=1, params=opts)
    cnm.debug_visualize = True
    cnm._debug_image = np.mean(Y, axis=0)
    wrapper = CNMFWrapper(cnmf_obj=cnm, mmap_path=mmap_path, dims=dims, output_dir=output_dir)
    wrapper.run()