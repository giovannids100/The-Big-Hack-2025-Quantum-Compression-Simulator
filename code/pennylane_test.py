import pennylane as qml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def prepare_image_data(image_path, target_size=(8, 8), use_color=False):

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
    img_array = np.array(img).flatten()
    
    print(f"Total values: {len(img_array)} ({'3 channels' if use_color else '1 channel'})")
    print(f"Qubits needed: {int(np.ceil(np.log2(len(img_array))))} qubits")
    
    # Normalize to unit vector (required for amplitude encoding)
    norm = np.linalg.norm(img_array)
    if norm > 0:
        img_array = img_array / norm
    
    return img_array, np.array(img), np.array(original_img.convert('RGB' if use_color else 'L'))

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

    print("="*70)
    print(f"ENCODING IMAGE: {os.path.basename(image_path)}")
    print("="*70)
    
    # Load and preprocess
    image_data, resized_img, original_img = prepare_image_data(image_path, target_size, use_color)
    
    # Encode in quantum circuit
    circuit, n_qubits = amplitude_encode_image(image_data)
    
    # Execute circuit
    print(f"\nExecuting quantum circuit...")
    quantum_state = circuit()
    
    print(f"✅ Encoding complete!")
    print(f"Quantum state shape: {quantum_state.shape}")
    
    # Verify encoding
    verify_encoding(image_data, quantum_state)
    
    # Visualize
    visualize_encoding(original_img, resized_img, image_data, quantum_state, n_qubits, use_color)
    
    return circuit, image_data, quantum_state

def visualize_encoding(original_img, resized_img, image_data, quantum_state, n_qubits, use_color=False):
    """
    Visualize the encoding process - simplified to show only before and after images.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Determine colormap
    cmap = None if use_color else 'gray'
    
    # Original image
    axes[0].imshow(original_img, cmap=cmap)
    axes[0].set_title(f'Original Image\n{original_img.shape[0]}×{original_img.shape[1]} pixels', fontsize=14)
    axes[0].axis('off')
    
    # Resized/encoded image
    axes[1].imshow(resized_img, cmap=cmap)
    channels_text = 'RGB' if use_color else 'Grayscale'
    axes[1].set_title(f'Quantum Encoded Image\n{resized_img.shape[0]}×{resized_img.shape[1]} pixels ({n_qubits} qubits)\n{channels_text}', fontsize=14)
    axes[1].axis('off')
    
    plt.suptitle('Quantum Image Encoding: Before & After', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":

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
    
    circuit, data, state = encode_image_from_file(image_path, target_size, use_color)
    