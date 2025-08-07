import os
import numpy as np
from tifffile import imread
import warnings
import sys
from contextlib import redirect_stderr
from io import StringIO

# Simple error suppression for YrA plotting issues
def suppress_YrA_errors():
    """Suppress common YrA visualization errors"""
    warnings.filterwarnings("ignore", message=".*cannot reshape array.*")
    warnings.filterwarnings("ignore", message=".*Failed to plot YrA.*")

suppress_YrA_errors()

from caiman.source_extraction.cnmf import cnmf, params
from caiman.mmapping import load_memmap, save_memmap

# Create a custom output filter for YrA messages only
class YrAErrorFilter:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        
    def write(self, text):
        # Only filter out specific YrA error messages, let everything else through
        if ("Failed to plot YrA" in text and "cannot reshape array" in text):
            # Replace YrA error with a cleaner message
            self.original_stream.write("⚠️ YrA visualization skipped (shape mismatch)\n")
        else:
            self.original_stream.write(text)
    
    def flush(self):
        self.original_stream.flush()
        
    def __getattr__(self, name):
        return getattr(self.original_stream, name)

# Manual debug saver that works without visualization
class ManualDebugSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_stage(self, cnmf_obj, stage_name):
        """Save CNMF outputs at each stage without problematic visualization"""
        print(f"📸 Saving stage: {stage_name}")
        
        def save_array(name, arr, stage):
            if arr is not None:
                try:
                    if hasattr(arr, 'toarray'):
                        arr = arr.toarray()
                    filename = os.path.join(self.output_dir, f"{name}_{stage}.npy")
                    np.save(filename, arr)
                    print(f"  ✅ Saved {name} shape {arr.shape} to {filename}")
                except Exception as e:
                    print(f"  ❌ Failed to save {name}: {e}")
        
        # Save all matrices
        if hasattr(cnmf_obj, 'estimates'):
            save_array("A", cnmf_obj.estimates.A, stage_name)
            save_array("C", cnmf_obj.estimates.C, stage_name)
            save_array("b", cnmf_obj.estimates.b, stage_name)
            save_array("f", cnmf_obj.estimates.f, stage_name)
            
            # Save YrA properly as temporal data (don't try to visualize as image)
            if hasattr(cnmf_obj.estimates, "YrA") and cnmf_obj.estimates.YrA is not None:
                save_array("YrA", cnmf_obj.estimates.YrA, stage_name)
                print(f"  📊 YrA shape: {cnmf_obj.estimates.YrA.shape} (K components × T timepoints)")
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, f"metadata_{stage_name}.txt")
        with open(metadata_path, "w") as f:
            f.write(f"Stage: {stage_name}\n")
            if hasattr(cnmf_obj, 'estimates'):
                f.write(f"A shape: {getattr(cnmf_obj.estimates.A, 'shape', 'None')}\n")
                f.write(f"C shape: {getattr(cnmf_obj.estimates.C, 'shape', 'None')}\n")
                f.write(f"YrA shape: {getattr(cnmf_obj.estimates.YrA, 'shape', 'None')}\n")
            f.write(f"dims: {getattr(cnmf_obj, 'dims', 'None')}\n")
        print(f"  📝 Saved metadata to {metadata_path}")

# ---------- Wrapper for CNMF Execution Without Manual Debug ----------

class CNMFWrapper:
    def __init__(self, cnmf_obj, mmap_path, dims, output_dir):
        self.cnmf = cnmf_obj
        self.mmap_path = mmap_path
        self.dims = dims
        self.debug_saver = ManualDebugSaver(output_dir)

    def run(self):
        # Filter YrA error messages during CNMF execution
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = YrAErrorFilter(original_stdout)
        sys.stderr = YrAErrorFilter(original_stderr)
        
        try:
            # Load memmap data correctly
            print(f"📖 Loading memmap from: {self.mmap_path}")
            
            # Try CaImAn's load_memmap first
            try:
                Yr, dims_loaded, T = load_memmap(self.mmap_path)
                dims = dims_loaded
                print(f"📖 Loaded with load_memmap: Yr shape {Yr.shape}, dims={dims}, T={T}")
            except:
                # Fallback: load directly as memmap
                print("📖 Using direct memmap loading...")
                T = self.dims[0] * self.dims[1]  # This will be corrected below
                
                # Load the memmap file directly
                fp_in = np.memmap(self.mmap_path, dtype=np.float32, mode='r')
                n_pixels = self.dims[0] * self.dims[1]
                T = fp_in.shape[0] // n_pixels
                
                Yr = fp_in.reshape((n_pixels, T), order='F')
                dims = self.dims
                print(f"📖 Direct load: Yr shape {Yr.shape}, dims={dims}, T={T}")
            
            self.cnmf.dims = self.dims
            
            # Ensure debug visualization is completely disabled
            self.cnmf.debug_visualize = False
            if hasattr(self.cnmf, '_debug_image'):
                delattr(self.cnmf, '_debug_image')
            
            # Check if only initialization
            only_init = self.cnmf.params.get('patch', 'only_init')
            if only_init is None:
                only_init = False
            print(f"🔧 Running CNMF with only_init = {only_init}")
            
            if only_init:
                # For only_init=True, just do initialization
                print("🚀 Starting initialization only...")
                self.cnmf.fit_file(self.mmap_path)
                self.debug_saver.save_stage(self.cnmf, "after_init_only")
            else:
                # Force manual step-by-step approach to save all debug stages
                print("🚀 Using manual step-by-step approach to save all debug stages...")
                
                # Manual approach for complete debugging
                # Load and preprocess data
                print("📖 Loading and preprocessing data...")
                Yr = self.cnmf.preprocess(Yr)
                
                # Initialization
                print("🎯 Running initialization...")
                self.cnmf.initialize(Yr.reshape((-1, T), order='F'))
                self.debug_saver.save_stage(self.cnmf, "after_initialize")
                
                # Spatial update 1
                print("🗺️ Running spatial update 1...")
                try:
                    self.cnmf.update_spatial(Yr.reshape((-1, T), order='F'))
                    self.debug_saver.save_stage(self.cnmf, "after_spatial_1")
                except (IndexError, ValueError) as e:
                    print(f"⚠️ Spatial update 1 failed: {e}")
                    print("⚠️ Skipping spatial update 1...")
                    self.debug_saver.save_stage(self.cnmf, "after_spatial_1_failed")
                
                # Temporal update 1
                print("⏰ Running temporal update 1...")
                try:
                    self.cnmf.update_temporal(Yr.reshape((-1, T), order='F'))
                    self.debug_saver.save_stage(self.cnmf, "after_temporal_1")
                except (IndexError, ValueError) as e:
                    print(f"⚠️ Temporal update 1 failed: {e}")
                    print("⚠️ Skipping temporal update 1...")
                    self.debug_saver.save_stage(self.cnmf, "after_temporal_1_failed")
                
                # Merging (if enabled)
                do_merge = self.cnmf.params.get('merging', 'do_merge')
                if do_merge is None:
                    do_merge = True
                if do_merge:
                    print("🔗 Running component merging...")
                    try:
                        self.cnmf.merge_comps(Yr.reshape((-1, T), order='F'))
                        self.debug_saver.save_stage(self.cnmf, "after_merge")
                    except (IndexError, ValueError) as e:
                        print(f"⚠️ Merging failed: {e}")
                        print("⚠️ Continuing without merging...")
                        # Save the stage anyway without merging
                        self.debug_saver.save_stage(self.cnmf, "after_merge_failed")
                
                # Spatial update 2
                print("🗺️ Running spatial update 2...")
                try:
                    self.cnmf.update_spatial(Yr.reshape((-1, T), order='F'))
                    self.debug_saver.save_stage(self.cnmf, "after_spatial_2")
                except (IndexError, ValueError) as e:
                    print(f"⚠️ Spatial update 2 failed: {e}")
                    print("⚠️ Skipping spatial update 2...")
                    self.debug_saver.save_stage(self.cnmf, "after_spatial_2_failed")
                
                # Temporal update 2
                print("⏰ Running temporal update 2...")
                try:
                    self.cnmf.update_temporal(Yr.reshape((-1, T), order='F'))
                    self.debug_saver.save_stage(self.cnmf, "after_temporal_2")
                except (IndexError, ValueError) as e:
                    print(f"⚠️ Temporal update 2 failed: {e}")
                    print("⚠️ Skipping temporal update 2...")
                    self.debug_saver.save_stage(self.cnmf, "after_temporal_2_failed")
                
                # Final residuals computation
                print("🧮 Computing final residuals...")
                try:
                    self.cnmf.compute_residuals(Yr.reshape((-1, T), order='F'))
                    self.debug_saver.save_stage(self.cnmf, "final")
                except (IndexError, ValueError) as e:
                    print(f"⚠️ Final residuals computation failed: {e}")
                    print("⚠️ Skipping residuals computation...")
                    self.debug_saver.save_stage(self.cnmf, "final_no_residuals")
            
            # Save final output
            output_path = os.path.join(os.path.dirname(self.mmap_path), "cnmf_final_output.hdf5")
            self.cnmf.save(output_path)
            print(f"💾 Saved final CNMF output to {output_path}")
            
            print("A shape:", self.cnmf.estimates.A.shape)
            print("C shape:", self.cnmf.estimates.C.shape)
            print("Number of ROIs:", self.cnmf.estimates.A.shape[1])
            A_final = self.cnmf.estimates.A.toarray()
            print("Non-zero in A:", np.count_nonzero(A_final))
            
        finally:
            # Restore original streams
            sys.stdout = original_stdout
            sys.stderr = original_stderr

# ---------- MAIN ----------
if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), '20250409_3_Glut_1mM', '20250409_3_Glut_1mM_TIF_VIDEO.TIF')
    Y = imread(file_path)
    #Y = Y[:300]  # Optional: limit number of frames
    T, d1, d2 = Y.shape
    dims = (d1, d2)
    
    print(f"📊 Data info: {T} frames, {d1}x{d2} pixels")
    print(f"📊 Data range: {Y.min()} to {Y.max()}")
    print(f"📊 Data type: {Y.dtype}")

    output_dir = "/Users/sheergalor/Library/CloudStorage/GoogleDrive-sheergalor@gmail.com/My Drive/cnmf_debug_outputs_try_1"
    os.makedirs(output_dir, exist_ok=True)

    mmap_base = f"glut_d1_{d1}_d2_{d2}_d3_1_order_C_frames_{T}"
    mmap_path = os.path.join(output_dir, mmap_base + '.mmap')
    npy_path = os.path.join(output_dir, mmap_base + '.npy')

    # Create memmap file compatible with CaImAn's load_memmap
    print("📝 Creating CaImAn-compatible memmap...")
    
    # Convert to the right format and save
    Y_reshaped = Y.astype(np.float32)
    
    # Create the memmap file directly in the expected format
    fname_new = os.path.join(output_dir, mmap_base + '.mmap')
    
    # Use numpy's memmap to create the file in the right format
    fp_out = np.memmap(fname_new, dtype=np.float32, mode='w+', 
                      shape=(np.prod(Y_reshaped.shape[1:]), Y_reshaped.shape[0]))
    
    # Reshape and transpose to get (pixels, time) format that CaImAn expects
    for t in range(Y_reshaped.shape[0]):
        fp_out[:, t] = Y_reshaped[t].flatten(order='F')
    
    fp_out.flush()
    del fp_out
    
    mmap_path = fname_new
    print(f"📁 Created memmap: {mmap_path}")

    # Biological parameters based on your imaging setup
    fr = 1.08                               # imaging rate in frames per second
    decay_time = 20                         # length of a typical transient in seconds 
    dxy = (1.243, 1.243)                    # spatial resolution in x and y in (um per pixel)
    cell_diameter = 10                      # in microns
    d_px = int(cell_diameter // dxy[0])     # convert microns to pixels

    # CNMF parameters for source extraction and deconvolution
    p = 1                                   # CHANGED: order of the autoregressive system (1 is more stable than 2)
    gnb = 2                                 # number of global background components
    merge_thr = 0.7                         # CHANGED: merging threshold, max correlation allowed (higher = less aggressive merging)
    bas_nonneg = True                       # enforce nonnegativity constraint on calcium traces
    rf = None                               # CHANGED: No patches - analyze full image at once
    stride_cnmf = None                      # CHANGED: No patches
    K = 200                                 # REDUCED: expect fewer components to avoid indexing issues
    gSig = np.array([0.5*d_px, 0.5*d_px])  # expected half-width of neurons in pixels
    gSiz = 2*gSig + 1                       # Gaussian kernel width and height
    method_init = 'greedy_roi'            # initialization method (more stable than corr_pnr)
    # Parameters required for corr_pnr initialization (keeping for reference)
    # min_corr = 0.8                          # minimum local correlation for a seeded pixel
    # min_pnr = 10                            # minimum peak-to-noise ratio for a seeded pixel
    ssub = 1                                # spatial subsampling during initialization
    tsub = 1                                # temporal subsampling during initialization

    # parameters for component evaluation
    min_SNR = 1.2               # LOWERED MORE: signal to noise ratio for more sensitivity
    rval_thr = 0.7              # LOWERED MORE: space correlation threshold for more sensitivity
    cnn_thr = 0.99              # threshold for CNN based classifier
    cnn_lowest = 0.1            # neurons with cnn probability lower than this value are rejected
    
    print(f"🔬 Biological parameters:")
    print(f"  Cell diameter: {cell_diameter} μm = {d_px} pixels")
    print(f"  No patches - analyzing full image")
    print(f"  K (total components): {K}")
    print(f"  gSig: {gSig}")

    # parameters dictionary
    parameter_dict = {
        'fnames': [mmap_path],  # Use mmap path instead of file_path
        'fr': fr,
        'dxy': dxy,
        'decay_time': decay_time,
        'p': p,
        'nb': gnb,
        'rf': rf,  # None for no patches
        'K': K,
        'gSig': gSig,
        'gSiz': gSiz,
        'stride': stride_cnmf,  # None for no patches
        'method_init': method_init,
        # 'min_corr': min_corr,  # Only needed for corr_pnr
        # 'min_pnr': min_pnr,    # Only needed for corr_pnr
        'rolling_sum': True,
        'only_init': False,  # Set to False to run full pipeline
        'ssub': ssub,
        'tsub': tsub,
        'merge_thr': merge_thr,
        'bas_nonneg': bas_nonneg,
        'min_SNR': min_SNR,
        'rval_thr': rval_thr,
        'use_cnn': False,  # Disable CNN to avoid errors
        'min_cnn_thr': cnn_thr,
        'cnn_lowest': cnn_lowest,
        'dims': dims,
        'is3D': False,
        'data_format': 'mmap',
        'n_pixels_per_process': dims[0] * dims[1],  # Full image: all pixels in one process
        'do_merge': True,
        'merge_thresh': merge_thr
    }

    # Create CNMFParams using the biological parameters
    opts = params.CNMFParams(params_dict=parameter_dict)
    
    print(f"🚀 CNMF Parameters loaded successfully!")

    cnm = cnmf.CNMF(n_processes=1, params=opts)
    cnm.debug_visualize = False  # Disable debug visualization to prevent YrA errors

    wrapper = CNMFWrapper(cnmf_obj=cnm, mmap_path=mmap_path, dims=dims, output_dir=output_dir)
    wrapper.run()
