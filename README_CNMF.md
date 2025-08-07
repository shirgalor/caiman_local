# CaImAn CNMF Analysis Project

This repository contains a customized CaImAn CNMF (Constrained Non-negative Matrix Factorization) analysis pipeline for calcium imaging data.

## 🚀 Quick Start with Google Colab

**Run this project in Google Colab without any local setup:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shirgalor/caiman_local/blob/main/CNMF_Colab_Runner.ipynb)

### What the Colab notebook includes:
- ✅ Complete CNMF analysis pipeline
- ✅ Custom error handling and debug stage saving
- ✅ Memory management for large datasets
- ✅ Comprehensive visualizations
- ✅ Google Drive integration for persistent storage
- ✅ All intermediate stages saved for napari viewer

## 📁 Project Structure

```
caiman/source_extraction/cnmf/
├── runner.py              # Main CNMF analysis script
├── napari_runner.py       # Interactive napari viewer with stage switching
└── 20250409_3_Glut_1mM/  # Example data directory
CNMF_Colab_Runner.ipynb    # Google Colab notebook
```

## 🔧 Custom Features

### Enhanced Error Handling
- **YrA Error Filtering**: Suppresses common visualization errors
- **Robust Spatial/Temporal Updates**: Continues processing even if individual steps fail
- **Step-by-step Debugging**: Saves intermediate results at each CNMF stage

### Debug Stage Saving
- `after_initialize`: Initial component detection
- `after_spatial_1`: First spatial refinement  
- `after_temporal_1`: First temporal refinement
- `after_merge`: Component merging results
- `after_spatial_2`: Second spatial refinement
- `after_temporal_2`: Second temporal refinement
- `final`: Complete analysis results

### Interactive Visualization
- **Napari Integration**: Interactive ROI analysis with stage switching
- **Keyboard Shortcuts**: Switch between CNMF stages (1-7 or Q,W,E,R,T,Y,U)
- **Real-time Analysis**: Click ROIs for detailed trace analysis

## 🎯 Optimized Parameters

The pipeline is optimized for:
- **Cell Type**: Neurons in calcium imaging
- **Image Size**: 512x512 pixels
- **Frame Rate**: 1.08 Hz
- **Cell Diameter**: ~10 μm
- **Initialization**: `greedy_roi` (more stable than `corr_pnr`)

## 💾 Memory Requirements

### Local Analysis:
- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores
- **Storage**: 10-20GB for debug files

### Google Colab:
- **Available**: 12-15GB RAM, multi-core CPU
- **Storage**: Unlimited via Google Drive
- **Runtime**: 2-4 hours for full analysis

## 🚀 Usage Options

### Option 1: Google Colab (Recommended)
1. Click the "Open in Colab" badge above
2. Upload your TIF video file
3. Run all cells sequentially
4. Download results or access via Google Drive

### Option 2: Local Installation
```bash
# Clone repository
git clone https://github.com/shirgalor/caiman_local.git
cd caiman_local

# Install CaImAn
pip install caiman[complete]

# Run analysis
cd caiman/source_extraction/cnmf
python runner.py
```

## 📊 Expected Outputs

- **HDF5 file**: Complete CNMF results
- **NumPy arrays**: A (spatial), C (temporal), YrA (residuals) matrices
- **Visualizations**: Spatial components, temporal traces, statistics
- **Debug stages**: All intermediate processing steps
- **Metadata**: Analysis parameters and component counts

## 🔍 Troubleshooting

### Common Issues:
- **Memory errors**: Use Colab or limit frames in `runner.py`
- **File not found**: Ensure video file is in correct directory
- **Stage switching not working**: Download all `.npy` files for local napari use

### Performance Tips:
- **Large datasets**: Limit to 600-800 frames for Colab
- **Local analysis**: Use SSD storage for faster I/O
- **Memory optimization**: Close other applications during analysis

## 📚 References

- **CaImAn**: [Giovannucci et al., eLife 2019](https://elifesciences.org/articles/38173)
- **CNMF Algorithm**: [Pnevmatikakis et al., Neuron 2016](https://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3)

## 📄 License

This project is based on CaImAn and follows the same licensing terms.
