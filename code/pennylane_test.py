import pennylane as qml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os



def load_image_from_file(image_path, target_size=(8, 8), use_color=False):
    """
    Load and prepare image data from file.
    
    Args:
        image_path: Path to image file
        target_size: Target size (width, height)
        use_color: True for RGB, False for grayscale
    
    Returns:
        tuple: (normalized_array, resized_image, original_image, original_norm)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path)
    original_img = img.copy()
    
    print(f"Original image size: {img.size}")
    print(f"Original mode: {img.mode}")
    print(f"Resizing to: {target_size}")
    
    if use_color:
        img = img.convert('RGB')
        print(f"Mode: COLOR (RGB)")
    else:
        img = img.convert('L')
        print(f"Mode: GRAYSCALE")
    
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    # Prepare normalized data
    normalized_array, original_norm = prepare_image_array(img_array)
    
    return normalized_array, img_array, np.array(original_img.convert('RGB' if use_color else 'L')), original_norm


def prepare_image_array(image_array):
    """
    Prepare image array for quantum encoding (flatten and normalize).
    
    Args:
        image_array: NumPy array of image (can be 2D or 3D)
    
    Returns:
        tuple: (normalized_1d_array, original_norm)
    """
    # Flatten and convert to float
    flat_array = image_array.flatten().astype(float)
    
    print(f"Total values: {len(flat_array)}")
    print(f"Qubits needed: {int(np.ceil(np.log2(len(flat_array))))} qubits")
    
    # Store original norm for reconstruction
    original_norm = np.linalg.norm(flat_array)
    
    # Normalize to unit vector
    if original_norm > 0:
        flat_array = flat_array / original_norm
    
    return flat_array, original_norm


def create_quantum_encoder(image_data):
    """
    Create a quantum circuit that encodes image data using amplitude encoding.
    
    Args:
        image_data: Normalized 1D array of image pixels
    
    Returns:
        tuple: (quantum circuit function, number of qubits, device)
    """
    n_qubits = int(np.ceil(np.log2(len(image_data))))
    
    # Check if feasible for simulation
    if n_qubits > 20:
        print(f"\n⚠️  WARNING: {n_qubits} qubits requires {2**n_qubits:,} amplitudes")
        print(f"   This may exceed available memory!")
        print(f"   Recommended maximum: 20 qubits (1M pixels)")
    
    # Pad data to nearest power of 2 if needed
    padded_size = 2**n_qubits
    if len(image_data) < padded_size:
        image_data = np.pad(image_data, (0, padded_size - len(image_data)))
    
    # Create device
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit():
        # Amplitude embedding
        qml.AmplitudeEmbedding(features=image_data, wires=range(n_qubits), normalize=True)
        return qml.state()
    
    return circuit, n_qubits, dev


def encode_image_data(image_data, original_norm=None):
    """
    Encode pre-prepared image data into quantum state.
    Use this when you already have a normalized array.
    
    Args:
        image_data: Normalized 1D array
        original_norm: Original normalization factor (optional, for reconstruction)
    
    Returns:
        tuple: (circuit, quantum_state, n_qubits, original_norm)
    """
    print(f"Creating quantum encoder for {len(image_data)} values...")
    
    # Create encoder
    circuit, n_qubits, dev = create_quantum_encoder(image_data)
    
    # Execute circuit
    print(f"Executing quantum circuit with {n_qubits} qubits...")
    quantum_state = circuit()
    
    print(f"✅ Encoding complete!")
    print(f"Quantum state shape: {quantum_state.shape}")
    
    return circuit, quantum_state, n_qubits, original_norm


def reconstruct_image_from_quantum_state(quantum_state, original_shape, use_color=False, original_norm=1.0):
    """
    Reconstruct the image from quantum state amplitudes.
    
    Args:
        quantum_state: The quantum state vector
        original_shape: Shape of the target image (height, width) or (height, width, channels)
        use_color: Whether this is a color image
        original_norm: The normalization factor used (to denormalize)
    
    Returns:
        Reconstructed image as numpy array
    """
    # Extract amplitudes from quantum state
    amplitudes = np.abs(quantum_state)
    
    # Calculate how many values we need
    if use_color:
        num_values = original_shape[0] * original_shape[1] * 3
    else:
        num_values = original_shape[0] * original_shape[1]
    
    # Take only the values we need (rest is padding)
    reconstructed_data = amplitudes[:num_values]
    
    # Denormalize
    reconstructed_data = reconstructed_data * original_norm
    
    # Reshape back to image format
    if use_color:
        reconstructed_img = reconstructed_data.reshape(original_shape[0], original_shape[1], 3)
    else:
        reconstructed_img = reconstructed_data.reshape(original_shape[0], original_shape[1])
    
    # Clip to valid pixel range [0, 255]
    reconstructed_img = np.clip(reconstructed_img, 0, 255)
    
    return reconstructed_img.astype(np.uint8)


def verify_encoding(original_data, quantum_state):
    """
    Verify that the quantum state amplitudes match the original data.
    
    Args:
        original_data: Original normalized data
        quantum_state: Quantum state from circuit execution
    
    Returns:
        Extracted amplitudes from quantum state
    """
    amplitudes = np.abs(quantum_state[:len(original_data)])
    correlation = np.dot(amplitudes, original_data)
    print(f"Correlation between original and encoded: {correlation:.6f}")
    return amplitudes


def calculate_reconstruction_quality(original_img, reconstructed_img):
    """
    Calculate MSE and PSNR metrics for reconstruction quality.
    
    Args:
        original_img: Original image array
        reconstructed_img: Reconstructed image array
    
    Returns:
        tuple: (mse, psnr)
    """
    mse = np.mean((original_img.astype(float) - reconstructed_img.astype(float))**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    return mse, psnr


def visualize_encoding(original_img, resized_img, quantum_state, n_qubits, use_color=False, original_norm=1.0):
    """
    Visualize the encoding process - shows original and quantum-reconstructed image.
    
    Args:
        original_img: Original image array
        resized_img: Resized image array
        quantum_state: Quantum state vector
        n_qubits: Number of qubits used
        use_color: True for RGB, False for grayscale
        original_norm: Normalization factor
    
    Returns:
        Reconstructed image
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    cmap = None if use_color else 'gray'
    
    # Original image
    axes[0].imshow(original_img, cmap=cmap)
    axes[0].set_title(f'Original Image\n{original_img.shape[0]}×{original_img.shape[1]} pixels', 
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Reconstruct image from quantum state
    reconstructed_img = reconstruct_image_from_quantum_state(
        quantum_state, 
        resized_img.shape, 
        use_color,
        original_norm
    )
    
    # Show reconstructed image
    axes[1].imshow(reconstructed_img, cmap=cmap)
    channels_text = 'RGB' if use_color else 'Grayscale'
    axes[1].set_title(f'Quantum Reconstructed\n{reconstructed_img.shape[0]}×{reconstructed_img.shape[1]} pixels ({n_qubits} qubits)\n{channels_text}', 
                     fontsize=14, fontweight='bold', color='green')
    axes[1].axis('off')
    
    # Calculate and show difference
    if use_color:
        diff = np.abs(resized_img.astype(float) - reconstructed_img.astype(float))
        diff_normalized = diff / 255.0
        axes[2].imshow(diff_normalized)
    else:
        diff = np.abs(resized_img.astype(float) - reconstructed_img.astype(float))
        axes[2].imshow(diff, cmap='hot', vmin=0, vmax=255)
    
    # Calculate error metrics
    mse, psnr = calculate_reconstruction_quality(resized_img, reconstructed_img)
    
    axes[2].set_title(f'Difference Map\nMSE: {mse:.2f}, PSNR: {psnr:.2f} dB', 
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('Quantum Image Encoding: Original → Quantum Circuit → Reconstructed', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print reconstruction quality
    print(f"\n{'='*70}")
    print("RECONSTRUCTION QUALITY")
    print(f"{'='*70}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
    if psnr > 40:
        print("Quality: ✅ EXCELLENT (nearly perfect)")
    elif psnr > 30:
        print("Quality: ✅ GOOD (high quality)")
    elif psnr > 20:
        print("Quality: ⚠️ FAIR (noticeable differences)")
    else:
        print("Quality: ❌ POOR (significant loss)")
    
    return reconstructed_img

def encode_image_from_file(image_path, target_size=(8, 8), use_color=False, visualize=True):
    """
    Complete pipeline: Load image from file and encode it.
    
    Args:
        image_path: Path to image file
        target_size: Target size (width, height)
        use_color: True for RGB, False for grayscale
        visualize: Whether to show visualization
    
    Returns:
        tuple: (circuit, image_data, quantum_state, reconstructed_image, original_norm)
    """
    print("="*70)
    print(f"ENCODING IMAGE: {os.path.basename(image_path)}")
    print("="*70)
    
    # Load and preprocess
    image_data, resized_img, original_img, original_norm = load_image_from_file(
        image_path, target_size, use_color
    )
    
    # Encode in quantum circuit
    circuit, quantum_state, n_qubits, _ = encode_image_data(image_data, original_norm)
    
    # Verify encoding
    verify_encoding(image_data, quantum_state)
    
    # Visualize
    reconstructed_img = None
    if visualize:
        reconstructed_img = visualize_encoding(
            original_img, resized_img, quantum_state, n_qubits, use_color, original_norm
        )
    
    return circuit, image_data, quantum_state, reconstructed_img, original_norm


def encode_image_from_array(image_array, use_color=False, visualize=False, original_norm=None):
    """
    Encode image from an existing NumPy array.
    Use this when you already have an image loaded in memory.
    
    Args:
        image_array: NumPy array (2D for grayscale, 3D for RGB)
        use_color: True if array is RGB
        visualize: Whether to show the image
        original_norm: If provided, use this norm instead of calculating
    
    Returns:
        tuple: (circuit, image_data, quantum_state, n_qubits, original_norm)
    """
    print("="*70)
    print(f"ENCODING IMAGE FROM ARRAY")
    print("="*70)
    
    # Prepare data
    image_data, calculated_norm = prepare_image_array(image_array)
    
    # Use provided norm or calculated norm
    norm_to_use = original_norm if original_norm is not None else calculated_norm
    
    # Encode
    circuit, quantum_state, n_qubits, _ = encode_image_data(image_data, norm_to_use)
    
    # Verify
    verify_encoding(image_data, quantum_state)
    
    # Optional visualization
    if visualize:
        cmap = None if use_color else 'gray'
        plt.figure(figsize=(6, 6))
        plt.imshow(image_array, cmap=cmap)
        plt.title(f'Encoded Image ({n_qubits} qubits)')
        plt.axis('off')
        plt.show()
    
    return circuit, image_data, quantum_state, n_qubits, norm_to_use


def encode_and_reconstruct_from_array(image_array, use_color=False, visualize=True):
    """
    Encode and immediately reconstruct an image from array.
    Useful for testing the full encode-decode pipeline.
    
    Args:
        image_array: NumPy array (2D for grayscale, 3D for RGB)
        use_color: True if array is RGB
        visualize: Whether to show visualization
    
    Returns:
        tuple: (circuit, quantum_state, reconstructed_image, original_norm)
    """
    print("="*70)
    print(f"ENCODING & RECONSTRUCTING IMAGE FROM ARRAY")
    print("="*70)
    
    # Prepare data
    image_data, original_norm = prepare_image_array(image_array)
    
    # Encode
    circuit, quantum_state, n_qubits, _ = encode_image_data(image_data, original_norm)
    
    # Verify
    verify_encoding(image_data, quantum_state)
    
    # Reconstruct
    reconstructed_img = reconstruct_image_from_quantum_state(
        quantum_state, image_array.shape, use_color, original_norm
    )
    
    # Visualize if requested
    if visualize:
        visualize_encoding(
            image_array, image_array, quantum_state, n_qubits, use_color, original_norm
        )
    
    return circuit, quantum_state, reconstructed_img, original_norm