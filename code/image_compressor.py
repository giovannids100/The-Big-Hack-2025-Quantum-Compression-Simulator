# Import necessary libraries
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize




def load_image(image_path, size=(64,64)):
    """Load an image, resize it, and convert it to a numpy array."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size)  # Increase resolution
        print(f"Image loaded successfully: {image_path}")
        return np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def amplitude_encode(image_array):
    """Amplitude encode the image data into a quantum circuit."""
    flat_img = image_array.flatten().astype(np.float32)
    norm_img = flat_img / np.linalg.norm(flat_img)

    num_amplitudes = len(norm_img)
    num_qubits = int(np.ceil(np.log2(num_amplitudes)))

    padded_img = np.pad(norm_img, (0, 2**num_qubits - num_amplitudes))

    circuit = QuantumCircuit(num_qubits)

    for i in range(2**num_qubits):
        theta = 2 * np.arccos(padded_img[i]) if padded_img[i] > 0 else 0
        circuit.ry(theta, i % num_qubits)

    print(f"Amplitude encoding completed for {len(padded_img)} amplitudes using {num_qubits} qubits.")
    return circuit, padded_img

def quantum_compression_technique_1(encoded_amplitudes):
    """Apply improved Quantum Compression Technique 1 (QCT1) to the encoded amplitudes."""
    num_qubits = int(np.ceil(np.log2(len(encoded_amplitudes))))
    circuit = QuantumCircuit(num_qubits)

    # Apply more sophisticated quantum gates for improved compression
    for i in range(num_qubits - 1):
        circuit.h(i)
        circuit.cx(i, i + 1)
        circuit.z(i + 1)
        circuit.cx(i, i + 1)
        circuit.h(i)

        
    print(circuit)
    simulator = Aer.get_backend('aer_simulator')
    transpiled_circuit = transpile(circuit, simulator)
    job = simulator.run(transpiled_circuit)
    result = job.result()

    # Implement a more gradual compression technique
    compression_factor = 0.75 # Adjust this value to control compression level
    compressed_length = int(len(encoded_amplitudes) * compression_factor)
    compressed_amplitudes = encoded_amplitudes[:compressed_length]
    compressed_amplitudes /= np.linalg.norm(compressed_amplitudes)

    print(f"Compression completed. Compressed amplitudes reduced to {len(compressed_amplitudes)}.")
    return compressed_amplitudes

def quantum_compression_technique_2(encoded_amplitudes):
    """Apply variational quantum compression (QCT2) using optimized parameterized circuits."""

    
    # Prepare amplitudes
    num_amplitudes = len(encoded_amplitudes)
    num_qubits = int(np.ceil(np.log2(num_amplitudes)))
    padded_length = 2**num_qubits
    
    # Pad amplitudes to power of 2
    padded_amplitudes = np.pad(encoded_amplitudes, (0, padded_length - num_amplitudes))
    padded_amplitudes = padded_amplitudes / np.linalg.norm(padded_amplitudes)
    
    # Define compression ratio
    compression_factor = 0.70  # More aggressive compression
    compressed_length = int(num_amplitudes * compression_factor)
    
    # Target state (what we want to preserve)
    target_state = padded_amplitudes[:compressed_length]
    target_state = target_state / np.linalg.norm(target_state)
    
    # Create variational quantum circuit
    def create_variational_circuit(params, n_qubits, n_layers=3):
        """Create a parameterized quantum circuit for compression."""
        circuit = QuantumCircuit(n_qubits)
        param_idx = 0
        
        for layer in range(n_layers):
            # Rotation layer
            for qubit in range(n_qubits):
                circuit.ry(params[param_idx], qubit)
                param_idx += 1
                circuit.rz(params[param_idx], qubit)
                param_idx += 1
            
            # Entanglement layer with controlled rotations
            for qubit in range(n_qubits - 1):
                circuit.cx(qubit, qubit + 1)
                circuit.crz(params[param_idx], qubit, qubit + 1)
                param_idx += 1
            
            # Add final entanglement between last and first qubit
            if n_qubits > 1:
                circuit.cx(n_qubits - 1, 0)
                circuit.crz(params[param_idx], n_qubits - 1, 0)
                param_idx += 1
        
        circuit.save_statevector()
        return circuit
    
    # Initialize parameters
    n_layers = 3
    n_params = n_layers * (2 * num_qubits + num_qubits)  # RY, RZ, and CRZ gates
    initial_params = np.random.uniform(-np.pi, np.pi, n_params)
    
    # Cost function for optimization
    backend = Aer.get_backend('aer_simulator')
    
    def cost_function(params):
        """Calculate fidelity loss between target and compressed state."""
        circuit = create_variational_circuit(params, num_qubits, n_layers)
        transpiled = transpile(circuit, backend)
        result = backend.run(transpiled, shots=1).result()
        statevector = result.get_statevector()
        
        # Extract and normalize compressed portion
        compressed_state = np.array(statevector)[:compressed_length]
        compressed_state = compressed_state / np.linalg.norm(compressed_state)
        
        # Calculate fidelity (overlap with target)
        fidelity = np.abs(np.vdot(target_state, compressed_state))**2
        
        return 1 - fidelity  # Minimize infidelity
    
    # Optimize circuit parameters
    print(f"Optimizing variational circuit for compression...")
    result = minimize(
       cost_function,
       initial_params,
       method='COBYLA',
       options={'maxiter': max(n_params + 50, 200), 'disp': False}
    )
    
    # Get final compressed amplitudes
    final_circuit = create_variational_circuit(result.x, num_qubits, n_layers)
    transpiled = transpile(final_circuit, backend)
    final_result = backend.run(transpiled, shots=1).result()
    final_statevector = final_result.get_statevector()
    
    compressed_amplitudes = np.array(final_statevector)[:compressed_length]
    compressed_amplitudes = compressed_amplitudes / np.linalg.norm(compressed_amplitudes)
    
    print(f"QCT2 compression completed. Reduced to {len(compressed_amplitudes)} amplitudes.")
    print(f"Final fidelity: {1 - result.fun:.4f}")
    
    return compressed_amplitudes

def visualize_images(original_image, encoded_amplitudes, compressed_amplitudes):
    """Visualize the original, encoded, and compressed images side by side."""
    compressed_image = reconstruct_image(compressed_amplitudes, original_image.shape)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')


    plt.subplot(1, 2, 2)
    plt.imshow(compressed_image)
    plt.title('Compressed Image (QCT1)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('quantum_image_processing_results.png', dpi=300)  # Save high-resolution image
    plt.show()

def visualize_compressed_image(original_image, compressed_amplitudes):
    compressed_image = reconstruct_image(compressed_amplitudes, original_image.shape)
    plt.imshow(compressed_image)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('quantum_compressed_image.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def reconstruct_image(amplitudes, original_shape):
    """Reconstruct an image from given amplitudes with improved quality."""
    num_pixels = np.prod(original_shape)

    if len(amplitudes) > num_pixels:
        truncated_amplitudes = amplitudes[:num_pixels]
    else:
        truncated_amplitudes = np.pad(amplitudes, (0, num_pixels - len(amplitudes)))

    reconstructed = truncated_amplitudes.reshape(original_shape)
    reconstructed = np.abs(reconstructed)

    # Apply contrast stretching for better visibility
    p2, p98 = np.percentile(reconstructed, (2, 98))
    reconstructed = np.clip(reconstructed, p2, p98)
    
    reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min()) * 255
    return reconstructed.astype(np.uint8)