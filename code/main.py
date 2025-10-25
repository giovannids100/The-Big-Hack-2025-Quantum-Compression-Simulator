import pennylane_test as pl
import video_splitter
import video_reconstructor

if __name__ == "__main__":

    video_path = r"./video.mp4"

    #change size to change resolution: size need to be 1x1 proportion and only 2^x numbers allowed
    frames = video_splitter.load_video_frames(video_path, size=(512,512), max_frames=None, frame_skip=1)
    #Higher resolutions require more computational power
    
    encoded_frames = []
    for frame in frames:
        circuit, data, state, qubits, norm = pl.encode_image_from_array(
            frame,
            use_color=True,
            visualize=False
        )
        encoded_frames.append({
            'state': state,
            'norm': norm,
            'shape': frame.shape
        })
    
    # Ricostruisci TUTTI i frame (questo mancava!)
    reconstructed_frames = []  # ← Lista per raccogliere tutti i frame
    for encoded in encoded_frames:
        reconstructed = pl.reconstruct_image_from_quantum_state(
            encoded['state'],
            encoded['shape'],
            use_color=True,
            original_norm=encoded['norm']
        )
        reconstructed_frames.append(reconstructed)
    
    # Ora passa la lista completa
    fps,tot_frames, is_reconstructed = video_reconstructor.reconstruct_video_with_original_fps(
        reconstructed_frames,
        "./compressed_video.mp4",
        "./video.mp4"
    )
    print(f"\n\nvideo reconstructed: compressed images size = {qubits} qubits, frames per secon = {fps}, video duration = {tot_frames/fps}s, total frames number = {tot_frames};\n total compressed video size = {qubits*tot_frames} qubits")
    