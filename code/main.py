import image_compressor as comp

if __name__ == "__main__":
    img_path = r"./image.png"  # Update with your image path
    img_array = comp.load_image(img_path, size=(1024, 1024))  # Increased resolution

    if img_array is not None:
        circuit, encoded_amplitudes = comp.amplitude_encode(img_array)
        compressed_amplitudes = comp.quantum_compression_technique_1(encoded_amplitudes)
        comp.visualize_images(img_array, encoded_amplitudes, compressed_amplitudes)
        visualize_compressed_image(img_array, compressed_amplitudes)
    else:
        print("Failed to load image, exiting the program.")