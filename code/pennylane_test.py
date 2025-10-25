import pennylane as qml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def prepare_image_data(image_path, target_size=(8, 8), use_color=False):
    """
    Load and preprocess an image for quantum encoding.
    
    Args:
        image_path: Path to image file
        target_size: Resize image to this size (width, height)
                    For amplitude encoding, total pixels should be ≤ 2^20 (~1M)
                    Recommended: (4,4)=16, (8,8)=64, (16,16)=256
        use_color: If True, preserve RGB channels. If False, convert to grayscale.
                   WARNING: Color images need 3x more qubits!
    
    Returns:
        Normalized flattened image array, original image, and normalization factor
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    img = Image.open(image_path)
    original_img = img.copy()
    
    print(f"Original image size: {img.size}")
    print(f"Original mode: {img.mode}")
    print(f"Resizing to: {target_size}")
    
    if use_color:
        # Convert to RGB (3 channels)
        img = img.convert('RGB')
        print(f"Mode: COLOR (RGB)")
    else:
        # Convert to grayscale (1 channel)
        img = img.convert('L')
        print(f"Mode: GRAYSCALE")
    
    # Resize to target size
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and flatten
    img_array = np.array(img).flatten().astype(float)
    
    print(f"Total values: {len(img_array)} ({'3 channels' if use_color else '1 channel'})")
    print(f"Qubits needed: {int(np.ceil(np.log2(len(img_array))))} qubits")
    
    # Store the original norm for reconstruction
    original_norm = np.linalg.norm(img_array)
    
    # Normalize to unit vector (required for amplitude encoding)
    if original_norm > 0:
        img_array = img_array / original_norm
    
    return img_array, np.array(img), np.array(original_img.convert('RGB' if use_color else 'L')), original_norm

def amplitude_encode_image(image_data):
    """
    Create a quantum circuit that encodes image data using amplitude encoding.
    
    Args:
        image_data: Normalized 1D array of image pixels
    
    Returns:
        Quantum circuit function and number of qubits
    """
    n_qubits = int(np.ceil(np.log2(len(image_data))))
    
    # Check if feasible for simulation
    if n_qubits > 20:
        print(f"\n WARNING: {n_qubits} qubits requires {2**n_qubits:,} amplitudes")
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
        # Amplitude embedding - encodes data in quantum state amplitudes
        qml.AmplitudeEmbedding(features=image_data, wires=range(n_qubits), normalize=True)
        
        # Return the state vector to verify encoding
        return qml.state()
    
    return circuit, n_qubits

def verify_encoding(original_data, quantum_state):
    """
    Verify that the quantum state amplitudes match the original data.
    """
    # Extract amplitudes from quantum state
    amplitudes = np.abs(quantum_state[:len(original_data)])
    
    # Compare with original (accounting for normalization)
    correlation = np.dot(amplitudes, original_data)
    print(f"Correlation between original and encoded: {correlation:.6f}")
    
    return amplitudes

def encode_image_from_file(image_path, target_size=(8, 8), use_color=False):
    """
    Complete pipeline: Load image from file and encode it.
    
    Args:
        image_path: Path to your image file (jpg, png, etc.)
        target_size: Tuple (width, height) for resizing
                    Examples: (4,4)=16px, (8,8)=64px, (16,16)=256px, (32,32)=1024px
        use_color: If True, encode RGB color channels (3x more qubits!)
                   If False, convert to grayscale (1 channel)
    
    Returns:
        circuit, image data, quantum state, and reconstructed image
    """
    print("="*70)
    print(f"ENCODING IMAGE: {os.path.basename(image_path)}")
    print("="*70)
    
    # Load and preprocess
    image_data, resized_img, original_img, original_norm = prepare_image_data(image_path, target_size, use_color)
    
    # Encode in quantum circuit
    circuit, n_qubits = amplitude_encode_image(image_data)
    
    # Execute circuit
    print(f"\nExecuting quantum circuit...")
    quantum_state = circuit()
    
    print(f"✅ Encoding complete!")
    print(f"Quantum state shape: {quantum_state.shape}")
    
    # Verify encoding
    verify_encoding(image_data, quantum_state)
    
    # Visualize with reconstruction
    reconstructed_img = visualize_encoding(original_img, resized_img, image_data, quantum_state, n_qubits, use_color, original_norm)
    
    return circuit, image_data, quantum_state, reconstructed_img

def reconstruct_image_from_quantum_state(quantum_state, original_shape, use_color=False, original_norm=1.0):
    """
    Reconstruct the image from quantum state amplitudes.
    This shows what the quantum circuit actually stores!
    
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
    
    # Denormalize (reverse the normalization we did earlier)
    reconstructed_data = reconstructed_data * original_norm
    
    # Reshape back to image format
    if use_color:
        reconstructed_img = reconstructed_data.reshape(original_shape[0], original_shape[1], 3)
    else:
        reconstructed_img = reconstructed_data.reshape(original_shape[0], original_shape[1])
    
    # Clip to valid pixel range [0, 255]
    reconstructed_img = np.clip(reconstructed_img, 0, 255)
    
    return reconstructed_img.astype(np.uint8)

def visualize_encoding(original_img, resized_img, image_data, quantum_state, n_qubits, use_color=False, original_norm=1.0):
    """
    Visualize the encoding process - shows original and quantum-reconstructed image.
    Now actually uses the quantum circuit to reconstruct the image!
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Determine colormap
    cmap = None if use_color else 'gray'
    
    # Original image
    axes[0].imshow(original_img, cmap=cmap)
    axes[0].set_title(f'Original Image\n{original_img.shape[0]}×{original_img.shape[1]} pixels', fontsize=14, fontweight='bold')
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
        # For color images, calculate difference per channel
        diff = np.abs(resized_img.astype(float) - reconstructed_img.astype(float))
        diff_normalized = diff / 255.0  # Normalize to [0,1] for visualization
        axes[2].imshow(diff_normalized)
    else:
        # For grayscale
        diff = np.abs(resized_img.astype(float) - reconstructed_img.astype(float))
        axes[2].imshow(diff, cmap='hot', vmin=0, vmax=255)
    
    # Calculate error metrics
    mse = np.mean((resized_img.astype(float) - reconstructed_img.astype(float))**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    axes[2].set_title(f'Difference Map\nMSE: {mse:.2f}, PSNR: {psnr:.2f} dB', fontsize=14, fontweight='bold')
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



if __name__ == "__main__":
    print("=== Quantum Amplitude Encoding for Images ===\n")

    print("\n" + "="*70)
    print("Encoding real image from file...")
    print("="*70)
    
    # Replace with your image path
    image_path = "./image.png"
    
    # Choose your target size - larger = more qubits needed!
    # (4,4)=16px, (8,8)=64px, (16,16)=256px, (32,32)=1024px
    target_size = (1024, 1024)
    
    # Set use_color=True for RGB images, False for grayscale
    use_color = True  # Set to False for grayscale
    
    circuit, data, state, reconstructed = encode_image_from_file(image_path, target_size, use_color)
