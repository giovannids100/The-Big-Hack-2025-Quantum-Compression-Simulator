import image_compressor as comp
import video_splitter
import video_reconstructor

if __name__ == "__main__":
    # img_path = r"./image.png"  # Update with your image path
    # img_array = comp.load_image(img_path, size=(1024, 1024))  # Increased resolution
'''
    if img_array is not None:
        circuit, encoded_amplitudes = comp.amplitude_encode(img_array)
        compressed_amplitudes = comp.quantum_compression_technique_1(encoded_amplitudes)
        comp.visualize_images(img_array, encoded_amplitudes, compressed_amplitudes)
        visualize_compressed_image(img_array, compressed_amplitudes)
    else:
    # #     print("Failed to load image, exiting the program.")
'''
    # if img_array is not None:
    #     circuit, encoded_amplitudes = comp.amplitude_encode(img_array)
    #     compressed_amplitudes = comp.quantum_compression_technique_1(encoded_amplitudes)
    #     comp.visualize_images(img_array, encoded_amplitudes, compressed_amplitudes)
    # else:
    #     print("Failed to load image, exiting the program.")

    video_path = r"./video.mp4"

    frames = video_splitter.load_video_frames(video_path, size=(2560, 1440), max_frames=None, frame_skip=1)
    compressed_frames = []


    for f in frames:   
        if f is not None:
            circuit, encoded_amplitudes = comp.amplitude_encode(f)
            compressed_amplitudes = comp.quantum_compression_technique_1(encoded_amplitudes)
            compressed_image = comp.reconstruct_image(compressed_amplitudes, f.shape)
            compressed_frames.append(compressed_image)
        else:
            print("Failed to load image, exiting the program.")
    
    video_reconstructor.reconstruct_video_with_original_fps(compressed_frames, "./compressed_video.mp4",video_path)
    