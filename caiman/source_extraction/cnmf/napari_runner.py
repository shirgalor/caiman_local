import os
import numpy as np
import napari
import glob
from imageio.v3 import imread
import matplotlib.pyplot as plt
from caiman.mmapping import load_memmap
from skimage.measure import label
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def create_pixel_aligned_mean_image(Yr, dims):
    """Create mean image with exact same pixel ordering as CaImAn uses"""
    print(f"Creating pixel-aligned mean image from Yr shape {Yr.shape}")
    print(f"Target dims: {dims}")
    
    # Yr is (n_pixels, n_frames) in Fortran order
    # Mean across time (axis=1) to get (n_pixels,)
    mean_flat = np.mean(Yr, axis=1)
    print(f"Mean flat shape: {mean_flat.shape}")
    
    # Reshape using Fortran order to match the same pixel indexing as A matrix
    mean_img = mean_flat.reshape(dims, order='F')
    print(f"Mean image final shape: {mean_img.shape}")
    
    return mean_img


def load_debug_stage(output_dir, stage_name, dims):
    """Load A, C, b, f from saved npy files for a given debug stage"""
    def load_array(name):
        path = os.path.join(output_dir, f"{name}_{stage_name}.npy")
        if os.path.exists(path):
            try:
                return np.load(path)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                return None
        else:
            print(f"File not found: {path}")
            return None

    A = load_array("A")
    C = load_array("C")
    b = load_array("b")
    f = load_array("f")
    YrA = load_array("YrA")

    if A is not None:
        try:
            # CaImAn uses Fortran ordering - A is (n_pixels, n_components)
            # We need to reshape to (d1, d2, n_components) using Fortran order
            # to maintain the same pixel indexing as the original video
            if len(A.shape) == 2:
                n_pixels, n_components = A.shape
                print(f"A matrix: {A.shape} -> reshaping to {dims} x {n_components}")
                
                # Ensure we have the right number of pixels
                if n_pixels != dims[0] * dims[1]:
                    print(f"Pixel mismatch: A has {n_pixels} pixels, expected {dims[0] * dims[1]}")
                    A = None
                else:
                    # Reshape using Fortran order to match CaImAn's pixel indexing
                    A = A.reshape((*dims, n_components), order='F')
            else:
                # Already reshaped
                A = A.reshape((*dims, -1), order='F')
            print(f"Loaded stage '{stage_name}': A shape {A.shape}, C shape {C.shape if C is not None else 'None'}")
        except Exception as e:
            print(f"Failed to reshape A for stage '{stage_name}': {e}")
            A = None

    return {
        'A': A,
        'C': C,
        'b': b,
        'f': f,
        'YrA': YrA,
    }


def launch_napari_viewer(mean_img, correlation_map, raw_video, Yr_matrix, output_dir):
    # Create viewer with scale bar disabled to prevent zoom-related errors
    viewer = napari.Viewer()
    
    # Disable scale bar overlay to prevent logarithm domain errors during zoom
    try:
        viewer.scale_bar.visible = False
    except AttributeError:
        pass  # Scale bar may not be available in all napari versions

    viewer.add_image(raw_video, name="Raw Video", colormap="gray", contrast_limits=[np.min(raw_video), np.max(raw_video)])
    mean_layer = viewer.add_image(mean_img, name="Mean Image", colormap="gray", blending="additive")
    corr_layer = viewer.add_image(correlation_map, name="Local Correlation", colormap="gray", blending="additive")

    debug_stages = [
        "after_initialize",
        "after_spatial_1", 
        "after_temporal_1",
        "after_merge",
        "after_spatial_2",
        "after_temporal_2",
        "final"
    ]
    d1, d2 = raw_video.shape[1:]
    dims = (d1, d2)

    # Load all stages data
    all_stages_data = {}
    for stage_name in debug_stages:
        loaded = load_debug_stage(output_dir, stage_name, dims)
        if loaded["A"] is not None and loaded["C"] is not None:
            all_stages_data[stage_name] = loaded
            print(f"Loaded stage '{stage_name}': {loaded['A'].shape[-1]} components")
        else:
            print(f"Skipping stage '{stage_name}': missing A or C.")

    if not all_stages_data:
        print("No valid stages found!")
        return

    # Start with the final stage
    current_stage = "final"
    if current_stage not in all_stages_data:
        current_stage = list(all_stages_data.keys())[-1]

    # Create initial label image
    def create_label_image(stage_name):
        A_spatial = all_stages_data[stage_name]["A"]
        
        print(f"Creating label image for stage '{stage_name}' with {A_spatial.shape[-1]} components...")
        
        # HYBRID APPROACH: Fast for most pixels, complete coverage for all ROI pixels
        threshold = 1e-10
        label_img = np.zeros(dims, dtype=np.uint16)
        
        # Step 1: Fast vectorized assignment for pixels with clear winners
        max_vals = np.max(A_spatial, axis=2)
        best_rois = np.argmax(A_spatial, axis=2)
        strong_mask = max_vals > threshold
        label_img[strong_mask] = best_rois[strong_mask] + 1
        
        # Step 2: Ensure ALL ROI pixels are labeled (fix overlapping regions)
        for j in range(A_spatial.shape[-1]):
            roi_mask = A_spatial[:, :, j] > threshold
            unlabeled_roi_pixels = roi_mask & (label_img == 0)
            
            # Assign ROI label to any unlabeled pixels that belong to this ROI
            label_img[unlabeled_roi_pixels] = j + 1
            
            # Also handle the case where a pixel might be better assigned to this ROI
            # but was assigned to another (for overlapping regions)
            current_assignment = label_img[roi_mask]
            pixels_in_roi = np.sum(roi_mask)
            if pixels_in_roi > 0:
                # Check if this ROI should "claim" some pixels from other ROIs
                for y, x in zip(*np.where(roi_mask)):
                    current_label = label_img[y, x]
                    if current_label > 0:
                        current_roi = current_label - 1
                        current_val = A_spatial[y, x, current_roi] if current_roi < A_spatial.shape[-1] else 0
                        this_roi_val = A_spatial[y, x, j]
                        
                        # If this ROI has significantly higher value, reassign
                        if this_roi_val > current_val * 1.1:  # 10% higher threshold
                            label_img[y, x] = j + 1
        
        unique_labels = np.unique(label_img[label_img > 0])
        total_roi_pixels = np.sum(A_spatial > threshold)
        labeled_pixels = np.count_nonzero(label_img)
        print(f"HYBRID creation: {len(unique_labels)} unique labels covering {labeled_pixels} pixels")
        print(f"Label coverage: {labeled_pixels/total_roi_pixels*100:.1f}% of all ROI pixels are labeled")
        
        # Verify each ROI has clickable pixels
        for j in range(A_spatial.shape[-1]):
            roi_pixels_total = np.sum(A_spatial[:, :, j] > threshold)
            roi_pixels_labeled = np.sum(label_img == j + 1)
            coverage = (roi_pixels_labeled / roi_pixels_total * 100) if roi_pixels_total > 0 else 0
            if coverage < 50:  # Warn if less than 50% coverage
                print(f"ROI {j}: {roi_pixels_labeled}/{roi_pixels_total} pixels labeled ({coverage:.1f}%)")
        
        return label_img

    # Add the interactive labels layer
    initial_label_img = create_label_image(current_stage)
    labels_layer = viewer.add_labels(
        initial_label_img, 
        name=f"ROIs {current_stage} ({all_stages_data[current_stage]['A'].shape[-1]} components)", 
        opacity=0.7
    )
    
    # Automatically select the ROIs layer so it's ready for interaction
    viewer.layers.selection.active = labels_layer
    print(f"ROIs layer automatically selected and ready for interaction")

    # Create callback function
    def show_trace(layer, event):
        stage_name = current_stage
        if stage_name not in all_stages_data:
            print(f"Stage {stage_name} not available")
            return
            
        stage_data = all_stages_data[stage_name]
        A_spatial_stage = stage_data["A"]
        C_temporal_stage = stage_data["C"]
        YrA_stage = stage_data["YrA"]
        
        print(f"Click detected on layer: {layer.name}")
        value = layer.get_value(event.position)
        print(f"Value at position: {value}")
        print(f"Click position: {event.position}")
        
        # Handle 3D click positions (t, y, x) by extracting last 2 dimensions
        if len(event.position) >= 2:
            click_y, click_x = int(event.position[-2]), int(event.position[-1])
            print(f"Extracted 2D position: ({click_y}, {click_x})")
        else:
            print(f"Invalid click position format: {event.position}")
            return
            
        if value is None or value == 0:
            # Fallback: find ROI at click location by checking spatial components directly
            print(f"Label value is 0, checking spatial components directly at ({click_y}, {click_x})")
            if 0 <= click_y < dims[0] and 0 <= click_x < dims[1]:
                # Find which ROI has the highest spatial value at this location
                max_val = 0
                best_roi = -1
                for j in range(A_spatial_stage.shape[-1]):
                    spatial_val = A_spatial_stage[click_y, click_x, j]
                    if spatial_val > max_val:
                        max_val = spatial_val
                        best_roi = j
                
                if best_roi >= 0 and max_val > 0:
                    roi_index = best_roi
                    print(f"Found ROI {roi_index} at click location with spatial value {max_val:.6f}")
                else:
                    print(f"No ROI found at click location")
                    return
            else:
                print(f"No ROI selected in stage {stage_name}.")
                return
        else:
            roi_index = int(value) - 1
            if roi_index >= C_temporal_stage.shape[0]:
                print(f"ROI {roi_index} out of range for stage {stage_name}.")
                return

        # Verify pixel alignment at click location
        if 0 <= click_y < dims[0] and 0 <= click_x < dims[1]:
            spatial_val = A_spatial_stage[click_y, click_x, roi_index]
            print(f"🎯 At click ({click_y}, {click_x}): spatial value = {spatial_val:.6f}")
        else:
            print(f"🎯 Click ({click_y}, {click_x}) is outside image bounds")

        print(f"✅ Selected ROI {roi_index} from stage {stage_name}")
                
        trace_c = C_temporal_stage[roi_index]
        mask = A_spatial_stage[:, :, roi_index] > 0
        pix_indices = np.where(mask.flatten(order='F'))[0]
        trace_y = Yr_matrix[pix_indices, :].mean(axis=0)
        
        print(f"📊 Frame count check:")
        print(f"  trace_c shape: {trace_c.shape} (frames: {len(trace_c)})")
        print(f"  trace_y shape: {trace_y.shape} (frames: {len(trace_y)})")
        print(f"  C_temporal shape: {C_temporal_stage.shape}")
        print(f"  Yr_matrix shape: {Yr_matrix.shape}")
        
        # Ensure frame counts match for correlation calculation
        min_frames = min(len(trace_c), len(trace_y))
        if len(trace_c) != len(trace_y):
            print(f"⚠️ Frame count mismatch! Trimming to {min_frames} frames")
            trace_c = trace_c[:min_frames]
            trace_y = trace_y[:min_frames]
            print(f"✅ After trimming: trace_c={len(trace_c)}, trace_y={len(trace_y)}")

        # Get YrA trace if available
        trace_YrA = None
        if YrA_stage is not None and roi_index < YrA_stage.shape[0]:
            trace_YrA = YrA_stage[roi_index]
            if trace_YrA is not None and len(trace_YrA) != len(trace_c):
                trace_YrA = trace_YrA[:min_frames]
                print(f"✅ Trimmed YrA trace to {len(trace_YrA)} frames")

        # Create organized matplotlib figure with better layout
        fig = plt.figure(figsize=(24, 12))
        fig.patch.set_facecolor('white')
        
        # Calculate correlation for display
        correlation = np.corrcoef(trace_y, trace_c)[0, 1]
        
        # Main title with key info
        fig.suptitle(f"CNMF Analysis | Stage: {stage_name} | ROI: {roi_index} | Correlation: {correlation:.3f}", 
                    fontsize=18, fontweight='bold', y=0.95)

        # TOP ROW: Spatial Information
        # 1. Original spatial component
        ax1 = plt.subplot(2, 4, 1)
        im1 = ax1.imshow(A_spatial_stage[:, :, roi_index], cmap='hot', interpolation='nearest')
        ax1.set_title(f"Spatial Component A[{roi_index}]", fontsize=14, fontweight='bold')
        ax1.set_xlabel("X pixels")
        ax1.set_ylabel("Y pixels")
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label("Spatial Weight", rotation=270, labelpad=15)

        # 2. A*C reconstruction mean
        ax2 = plt.subplot(2, 4, 2)
        ac_product = np.outer(A_spatial_stage[:, :, roi_index].flatten(order='F'), trace_c)
        ac_product = ac_product.reshape((A_spatial_stage.shape[0], A_spatial_stage.shape[1], -1), order='F')
        im2 = ax2.imshow(np.mean(ac_product, axis=2), cmap='viridis', interpolation='nearest')
        ax2.set_title("Mean A×C Reconstruction", fontsize=14, fontweight='bold')
        ax2.set_xlabel("X pixels")
        ax2.set_ylabel("Y pixels")
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label("Intensity", rotation=270, labelpad=15)

        # 3. Temporal traces (main plot)
        ax3 = plt.subplot(2, 4, (3, 4))
        time_frames = np.arange(len(trace_c))
        ax3.plot(time_frames, trace_y, label='Raw Data (Y)', color='#2E86C1', linewidth=2.5, alpha=0.8)
        ax3.plot(time_frames, trace_c, label='CNMF (C)', color='#E74C3C', linewidth=2.5)
        if trace_YrA is not None:
            ax3.plot(time_frames, trace_YrA, label='Residual (YrA)', color='#8E44AD', linewidth=1.5, alpha=0.7, linestyle='--')
        ax3.set_title("Temporal Activity Traces", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Frame Number", fontsize=12)
        ax3.set_ylabel("Fluorescence Intensity", fontsize=12)
        ax3.legend(loc='upper right', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # BOTTOM ROW: Analysis and Statistics
        # 4. Correlation scatter plot
        ax4 = plt.subplot(2, 4, 5)
        scatter = ax4.scatter(trace_y, trace_c, alpha=0.6, c=time_frames, cmap='viridis', s=30)
        ax4.set_xlabel("Raw Data (Y)", fontsize=12)
        ax4.set_ylabel("CNMF (C)", fontsize=12)
        ax4.set_title(f"Y vs C Correlation: {correlation:.3f}", fontsize=14, fontweight='bold')
        
        # Add correlation line
        z = np.polyfit(trace_y, trace_c, 1)
        p = np.poly1d(z)
        ax4.plot(trace_y, p(trace_y), "r--", alpha=0.8, linewidth=2)
        
        # Add colorbar for time
        cbar4 = plt.colorbar(scatter, ax=ax4, shrink=0.8)
        cbar4.set_label("Frame", rotation=270, labelpad=15)
        ax4.grid(True, alpha=0.3)

        # 5. Signal quality metrics
        ax5 = plt.subplot(2, 4, 6)
        
        # Calculate additional metrics
        snr_estimate = np.mean(trace_c) / np.std(trace_c) if np.std(trace_c) > 0 else 0
        peak_to_baseline = (np.max(trace_c) - np.min(trace_c)) / np.mean(trace_c) if np.mean(trace_c) > 0 else 0
        
        metrics = ['Correlation', 'SNR Est.', 'Peak/Baseline', 'Spatial Max']
        values = [correlation, snr_estimate, peak_to_baseline, np.max(A_spatial_stage[:, :, roi_index])]
        colors = ['#E74C3C', '#3498DB', '#F39C12', '#27AE60']
        
        bars = ax5.bar(metrics, values, color=colors, alpha=0.8)
        ax5.set_title("Quality Metrics", fontsize=14, fontweight='bold')
        ax5.set_ylabel("Value", fontsize=12)
        ax5.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # 6. Detailed statistics table
        ax6 = plt.subplot(2, 4, 7)
        ax6.axis('off')
        
        # Create organized stats table
        stats_data = [
            ["Stage", stage_name],
            ["ROI Index", f"{roi_index}"],
            ["Total Components", f"{A_spatial_stage.shape[-1]}"],
            ["ROI Pixels", f"{np.count_nonzero(mask)}"],
            ["Frames Used", f"{min_frames}"],
            ["C Max/Mean", f"{np.max(trace_c):.3f} / {np.mean(trace_c):.3f}"],
            ["Spatial Max", f"{np.max(A_spatial_stage[:, :, roi_index]):.3f}"],
            ["Y-C Correlation", f"{correlation:.3f}"],
        ]
        
        if trace_YrA is not None:
            stats_data.extend([
                ["YrA Max/Mean", f"{np.max(trace_YrA):.3f} / {np.mean(trace_YrA):.3f}"],
            ])
        
        # Create table
        table_text = ""
        for i, (label, value) in enumerate(stats_data):
            table_text += f"{label:.<20} {value}\n"
        
        ax6.text(0.05, 0.95, table_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#DEE2E6", linewidth=1))
        
        ax6.set_title("Component Statistics", fontsize=14, fontweight='bold')

        # 7. Activity histogram
        ax7 = plt.subplot(2, 4, 8)
        ax7.hist(trace_c, bins=30, alpha=0.7, color='#E74C3C', edgecolor='black', linewidth=0.5)
        ax7.axvline(np.mean(trace_c), color='#2E86C1', linestyle='--', linewidth=2, label=f'Mean: {np.mean(trace_c):.3f}')
        ax7.axvline(np.median(trace_c), color='#F39C12', linestyle='--', linewidth=2, label=f'Median: {np.median(trace_c):.3f}')
        ax7.set_title("Activity Distribution", fontsize=14, fontweight='bold')
        ax7.set_xlabel("Fluorescence", fontsize=12)
        ax7.set_ylabel("Frequency", fontsize=12)
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        plt.show()

    # Use napari's built-in key binding for ROI selection
    @viewer.bind_key('space')
    def analyze_roi(viewer):
        """Analyze ROI at current cursor position when space is pressed"""
        try:
            # Find and automatically select the ROIs layer
            roi_layer = None
            for layer in viewer.layers:
                if layer.name.startswith("ROIs"):
                    roi_layer = layer
                    break
            
            if roi_layer is None:
                print("❌ No ROIs layer found")
                return
            
            # Automatically select the ROIs layer and make it visible
            viewer.layers.selection.active = roi_layer
            roi_layer.visible = True
            print(f"✅ Auto-selected and activated layer: {roi_layer.name}")
            
            # Get cursor position
            cursor_pos = viewer.cursor.position
            print(f"🖱️ Analyzing ROI at cursor position: {cursor_pos}")
            
            # Create a mock event for the show_trace function
            class MockEvent:
                def __init__(self, position):
                    self.position = position
            
            mock_event = MockEvent(cursor_pos)
            show_trace(roi_layer, mock_event)
            
        except Exception as e:
            print(f"❌ Error analyzing ROI: {e}")
            print("💡 Hover over an ROI pixel and press SPACE to analyze")

    # Also add a simpler keyboard shortcut to show current position info
    @viewer.bind_key('i')
    def show_info(viewer):
        """Show pixel info at current cursor position when 'i' is pressed"""
        try:
            cursor_pos = viewer.cursor.position
            print(f"\n📍 Cursor position: {cursor_pos}")
            
            # Get all layer values at this position
            for layer in viewer.layers:
                if hasattr(layer, 'data'):
                    try:
                        value = layer.get_value(cursor_pos)
                        print(f"   {layer.name}: {value}")
                    except:
                        pass
                        
        except Exception as e:
            print(f"❌ Error showing info: {e}")

    # Add general click handler for all layers to show pixel values and alignment info
    def show_pixel_info(layer, event):
        """Show pixel information for any layer click"""
        print(f"\n🖱️ Click on layer: {layer.name}")
        print(f"🔍 Click position: {event.position}")
        
        # Handle 3D click positions by extracting last 2 dimensions
        if len(event.position) >= 2:
            click_y, click_x = int(event.position[-2]), int(event.position[-1])
            print(f"🔍 Extracted 2D position: ({click_y}, {click_x})")
            
            if 0 <= click_y < dims[0] and 0 <= click_x < dims[1]:
                # Show pixel values from all layers at this location
                if layer.name == "Raw Video":
                    if len(event.position) >= 3:
                        frame_idx = int(event.position[0])
                        if 0 <= frame_idx < raw_video.shape[0]:
                            pixel_val = raw_video[frame_idx, click_y, click_x]
                            print(f"📊 Raw video frame {frame_idx} at ({click_y}, {click_x}): {pixel_val}")
                    
                elif layer.name == "Mean Image":
                    pixel_val = mean_img[click_y, click_x]
                    print(f"📊 Mean image at ({click_y}, {click_x}): {pixel_val:.3f}")
                    
                elif layer.name == "Local Correlation":
                    pixel_val = correlation_map[click_y, click_x]
                    print(f"📊 Correlation map at ({click_y}, {click_x}): {pixel_val:.3f}")
                
                # Show spatial component values at this location for current stage
                if current_stage in all_stages_data:
                    A_spatial = all_stages_data[current_stage]["A"]
                    print(f"🎯 Spatial components at ({click_y}, {click_x}) for stage '{current_stage}':")
                    
                    # Find top 5 spatial components at this location
                    spatial_vals = []
                    for j in range(A_spatial.shape[-1]):
                        val = A_spatial[click_y, click_x, j]
                        if val > 0:
                            spatial_vals.append((j, val))
                    
                    spatial_vals.sort(key=lambda x: x[1], reverse=True)
                    top_components = spatial_vals[:5]
                    
                    if top_components:
                        print(f"   Top components: {[(f'ROI {roi}', f'{val:.6f}') for roi, val in top_components]}")
                    else:
                        print(f"   No spatial components found at this location")
                
            else:
                print(f"❌ Click ({click_y}, {click_x}) is outside image bounds")
    
    # Click handlers are now handled by the viewer-level callback above
    # No need for individual layer callbacks

    # Add keyboard shortcuts to switch between stages using different binding approach
    def switch_stage(stage_name):
        nonlocal current_stage
        if stage_name in all_stages_data:
            current_stage = stage_name
            update_layer()
        else:
            print(f"❌ Stage {stage_name} not available")

    @viewer.bind_key('q')
    def switch_to_initialize(viewer):
        switch_stage("after_initialize")

    @viewer.bind_key('w') 
    def switch_to_spatial1(viewer):
        switch_stage("after_spatial_1")

    @viewer.bind_key('e')
    def switch_to_temporal1(viewer):
        switch_stage("after_temporal_1")

    @viewer.bind_key('r')
    def switch_to_merge(viewer):
        switch_stage("after_merge")

    @viewer.bind_key('t')
    def switch_to_spatial2(viewer):
        switch_stage("after_spatial_2")

    @viewer.bind_key('y')
    def switch_to_temporal2(viewer):
        switch_stage("after_temporal_2")

    @viewer.bind_key('u')
    def switch_to_final(viewer):
        switch_stage("final")

    # Also try the number keys with explicit key codes
    @viewer.bind_key('1')
    def key_1(viewer):
        switch_stage("after_initialize")

    @viewer.bind_key('2')
    def key_2(viewer):
        switch_stage("after_spatial_1")

    @viewer.bind_key('3')
    def key_3(viewer):
        switch_stage("after_temporal_1")

    @viewer.bind_key('4')
    def key_4(viewer):
        switch_stage("after_merge")

    @viewer.bind_key('5')
    def key_5(viewer):
        switch_stage("after_spatial_2")

    @viewer.bind_key('6')
    def key_6(viewer):
        switch_stage("after_temporal_2")

    @viewer.bind_key('7')
    def key_7(viewer):
        switch_stage("final")

    def update_layer():
        """Update the labels layer with the current stage"""
        print(f"🔄 Switching to stage: {current_stage}...")
        
        # Fast label image creation
        new_label_img = create_label_image(current_stage)
        labels_layer.data = new_label_img
        labels_layer.name = f"ROIs {current_stage} ({all_stages_data[current_stage]['A'].shape[-1]} components)"
        
        # Automatically select the ROIs layer after stage switch
        viewer.layers.selection.active = labels_layer
        
        print(f"✅ Switched to stage: {current_stage}")
        print(f"✅ ROIs layer automatically selected and ready")

    print("🎮 KEYBOARD SHORTCUTS to switch between CNMF stages:")
    print("  Number keys: 1-7 OR Letter keys: Q,W,E,R,T,Y,U")
    print("  1 or Q: after_initialize (400 components)")
    print("  2 or W: after_spatial_1 (385 components)") 
    print("  3 or E: after_temporal_1 (385 components)")
    print("  4 or R: after_merge (385 components)")
    print("  5 or T: after_spatial_2 (360 components)")
    print("  6 or Y: after_temporal_2 (360 components)")
    print("  7 or U: final (360 components)")
    print("� INTERACTION SHORTCUTS:")
    print("  �💡 HOVER over ROI pixel + press SPACE → detailed trace analysis")
    print("  💡 HOVER over any pixel + press 'I' → pixel values and alignment info")
    print("✅ ROI layer is AUTOMATICALLY selected (no manual selection needed!)")
    print("⚠️  Make sure napari window is focused before pressing keys!")

    napari.run()


if __name__ == "__main__":
    # Update to match your output directory and file structure
    output_dir = "/Users/sheergalor/Library/CloudStorage/GoogleDrive-sheergalor@gmail.com/My Drive/cnmf_debug_outputs_try_1"

    # Try to find the most recent mmap file
    import glob
    mmap_files = glob.glob(os.path.join(output_dir, "glut_d1_512_d2_512_d3_1_order_C_frames_*.mmap"))
    if mmap_files:
        # Use the most recent one (largest frame count)
        mmap_path = max(mmap_files, key=lambda x: int(x.split('_frames_')[1].split('.')[0]))
        print(f"📁 Found mmap file: {os.path.basename(mmap_path)}")
    else:
        print("❌ No mmap files found")
        mmap_path = None
    
    try:
        if mmap_path and os.path.exists(mmap_path):
            Yr, dims, T = load_memmap(mmap_path)
            print(f"📖 Loaded memmap: {Yr.shape}, dims: {dims}, T: {T}")
            
            # Create pixel-aligned mean image using same ordering as CaImAn
            mean_img = create_pixel_aligned_mean_image(Yr, dims)
            
            # Create a simple correlation map (or use mean as placeholder)
            correlation_map = mean_img.copy()
        else:
            raise FileNotFoundError("No valid mmap file found")
        
    except Exception as e:
        print(f"❌ Error loading memmap: {e}")
        print("📝 Creating dummy images...")
        dims = (512, 512)
        mean_img = np.random.rand(*dims)
        correlation_map = np.random.rand(*dims)
        Yr = np.random.rand(dims[0] * dims[1], 100)  # Dummy data

    # Load raw video
    video_path = os.path.join(os.path.dirname(__file__), '20250409_3_Glut_1mM', '20250409_3_Glut_1mM_TIF_VIDEO.TIF')
    raw_video = imread(video_path)
    
    # Match the number of frames used in the mmap
    if mmap_path:
        frame_count = int(os.path.basename(mmap_path).split('_frames_')[1].split('.')[0])
        raw_video = raw_video[:frame_count]
        print(f"📹 Using {frame_count} frames from raw video")
    else:
        raw_video = raw_video[:100]  # Default fallback
    
    # Verify dimensions match for alignment check
    print(f"🔍 Alignment verification:")
    print(f"  Raw video: {raw_video.shape}")
    print(f"  Mmap Yr: {Yr.shape}")
    print(f"  Mean image: {mean_img.shape}")
    print(f"  Expected dims: {dims}")
    
    print(f"🚀 Launching Napari viewer for all debug stages...")
    launch_napari_viewer(mean_img, correlation_map, raw_video, Yr, output_dir)
