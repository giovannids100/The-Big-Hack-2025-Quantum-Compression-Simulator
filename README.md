# THE BIG HACK
# 🎬 Quantum Video Compression Simulator

A quantum computing-based video compression tool that uses **amplitude encoding** and **quantum circuits** to compress and reconstruct video files. This project leverages both **Qiskit** and **PennyLane** quantum frameworks to demonstrate quantum information processing on multimedia data.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-purple.svg)
![PennyLane](https://img.shields.io/badge/PennyLane-0.33+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🌟 Features

- **Quantum Amplitude Encoding**: Encodes image/video data into quantum states
- **Frame-by-Frame Processing**: Processes video frames independently for scalability
- **Reconstruction Pipeline**: Fully reconstructs videos from quantum states
- **Quality Metrics**: Calculates MSE and PSNR for compression quality analysis
- **Flexible Resolution**: Supports various frame sizes (recommended: 16×16 to 256×256)
- **RGB & Grayscale Support**: Works with both color and monochrome videos

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Usage Examples](#-usage-examples)
- [API Reference](#-api-reference)
- [Performance & Limitations](#-performance--limitations)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r code/requirements.txt
   ```

## ⚡ Quick Start

### Compress a Video

### Module Descriptions

| Module | Description |
|--------|-------------|
| `pennylane_test.py` | PennyLane implementation with modular encoding/decoding functions |
| `video_splitter.py` | Loads and extracts frames from video files |
| `video_reconstructor.py` | Reconstructs MP4 videos from frame arrays |
| `main.py` | Complete pipeline demonstration |

---

## 🔬 How It Works

### 1. **Amplitude Encoding**

The system encodes pixel values into quantum state amplitudes:

```
|ψ⟩ = Σᵢ αᵢ|i⟩
```

Where `αᵢ` represents normalized pixel values.

**Example**: A 16×16 RGB image (768 pixels) requires **10 qubits** (2¹⁰ = 1024 states).

### 2. **Quantum Circuit Processing**

The quantum circuit applies transformations:

- **Hadamard Gates (H)**: Create superposition
- **CNOT Gates (CX)**: Entangle qubits
- **Pauli-Z Gates (Z)**: Phase manipulation
- **Rotation Gates (RY)**: Amplitude adjustments

### 3. **Compression Techniques**

#### QCT1 (Quantum Compression Technique 1)
- Threshold-based amplitude selection
- Keeps only most significant quantum states
- Configurable compression ratio (default: 75%)

#### QCT2 (Qubit Tracing)
- Discards least significant qubits
- Reduces Hilbert space dimensionality
- True quantum compression via partial trace

### 4. **Reconstruction**

Quantum states are measured and converted back to pixel values with denormalization and clipping to valid range [0, 255].

---

## 📚 API Reference

### `pennylane_test.py`

#### `encode_image_from_file(image_path, target_size, use_color, visualize)`
Load and encode an image from file.

**Parameters:**
- `image_path` (str): Path to image file
- `target_size` (tuple): Target resolution (width, height)
- `use_color` (bool): RGB if True, grayscale if False
- `visualize` (bool): Show visualization plots

**Returns:** `(circuit, image_data, quantum_state, reconstructed_image, original_norm)`

#### `encode_image_from_array(image_array, use_color, visualize, original_norm)`
Encode image from NumPy array.

**Parameters:**
- `image_array` (ndarray): Image as NumPy array
- `use_color` (bool): Color mode
- `visualize` (bool): Show plots
- `original_norm` (float, optional): Pre-calculated normalization factor

**Returns:** `(circuit, image_data, quantum_state, n_qubits, original_norm)`

#### `reconstruct_image_from_quantum_state(quantum_state, original_shape, use_color, original_norm)`
Reconstruct image from quantum state.

**Parameters:**
- `quantum_state` (ndarray): Quantum state vector
- `original_shape` (tuple): Target image shape
- `use_color` (bool): Color mode
- `original_norm` (float): Denormalization factor

**Returns:** `ndarray` (reconstructed image)

### `video_splitter.py`

#### `load_video_frames(video_path, size, max_frames, frame_skip)`
Extract frames from video file.

**Parameters:**
- `video_path` (str): Path to MP4 video
- `size` (tuple): Target frame size
- `max_frames` (int, optional): Maximum frames to extract
- `frame_skip` (int): Extract every N-th frame

**Returns:** `list[ndarray]` (list of frames)

### `video_reconstructor.py`

#### `reconstruct_video(frames_list, output_path, fps, codec)`
Create video from frame list.

**Parameters:**
- `frames_list` (list): List of frame arrays
- `output_path` (str): Output video path
- `fps` (float): Frames per second
- `codec` (str): Video codec ('mp4v', 'avc1', 'H264')

**Returns:** `bool` (success status)

#### `reconstruct_video_with_original_fps(frames_list, output_path, original_video_path)`
Reconstruct video using original FPS.

---

## ⚠️ Performance & Limitations

### Qubit Requirements

| Resolution | Pixels | Qubits (Grayscale) | Qubits (RGB) |
|------------|--------|-------------------|--------------|
| 8×8        | 64     | 6                 | 9            |
| 16×16      | 256    | 8                 | 10           |
| 32×32      | 1,024  | 10                | 12           |
| 64×64      | 4,096  | 12                | 14           |
| 128×128    | 16,384 | 14                | 16           |
| 256×256    | 65,536 | 16                | 18           |

### Recommended Settings

- **Small videos**: 32×32, RGB, all frames
- **Medium videos**: 16×16, RGB, frame_skip=2
- **Large videos**: 16×16, grayscale, frame_skip=5
- **Maximum practical**: 256×256 (requires ~70GB RAM)

### Current Limitations

1. **Memory Intensive**: Quantum simulation requires exponential memory (2ⁿ states)
2. **Slow Processing**: Each frame takes several seconds to encode/decode
3. **No True Compression**: This is a *simulator* - actual quantum computers needed for real compression
4. **Quality Loss**: Some quality degradation due to normalization and amplitude truncation

### Performance Tips

```python
# ✅ GOOD: Reasonable for simulation
frames = load_video_frames("video.mp4", size=(16, 16), frame_skip=2)

# ⚠️ SLOW: Will take a long time
frames = load_video_frames("video.mp4", size=(64, 64), frame_skip=1)

# ❌ BAD: May crash with out-of-memory
frames = load_video_frames("video.mp4", size=(256, 256))
```

---

**Shorter videos** can be encoded in higher resolutions without too much computational effort

But using more than **20 qubits** can make the program crash!!!!

## 🎯 Future Improvements

- [ ] Hardware quantum computer integration (IBM Quantum, IonQ)
- [ ] Hybrid classical-quantum compression algorithms
- [ ] GPU-accelerated quantum simulation
- [ ] Real-time video processing
- [ ] Additional compression techniques (QJPEG, QPEG)
- [ ] Batch processing optimization
- [ ] Web interface for easy use


## 🌐 Resources

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [PennyLane Documentation](https://pennylane.ai/)
- [Quantum Amplitude Encoding Paper](https://arxiv.org/abs/quant-ph/0407010)
- [Quantum Image Processing Survey](https://arxiv.org/abs/1801.01465)

---

<div align="center">
  
**⭐ Star this repository if you find it useful!**

Made with ❤️ and ⚛️ (quantum computing)

</div>
