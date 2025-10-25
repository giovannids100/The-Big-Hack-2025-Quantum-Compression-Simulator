import cv2
import numpy as np


def reconstruct_video(frames_list, output_path, fps=30, codec='mp4v'):
    """
    Reconstruct a video file from a list of frame arrays.
    
    Parameters:
    -----------
    frames_list : list of numpy arrays
        List of frames as numpy arrays (height, width, channels)
        Each frame should be uint8 with values 0-255
    output_path : str
        Path where the output video will be saved (e.g., 'output.mp4')
    fps : int or float
        Frames per second for the output video (default: 30)
    codec : str
        FourCC codec code (default: 'mp4v' for MP4)
        Other options: 'XVID', 'H264', 'avc1'
    
    Returns:
    --------
    bool
        True if video was created successfully, False otherwise
    """
    if not frames_list or len(frames_list) == 0:
        print("Error: frames_list is empty")
        return False
    
    try:
        # Get dimensions from first frame
        first_frame = frames_list[0]
        height, width = first_frame.shape[:2]
        
        # Check if frames are grayscale or color
        is_color = len(first_frame.shape) == 3 and first_frame.shape[2] == 3
        
        print(f"Reconstructing video:")
        print(f"  Resolution: {width}x{height}")
        print(f"  Total frames: {len(frames_list)}")
        print(f"  FPS: {fps}")
        print(f"  Color: {'Yes' if is_color else 'No (Grayscale)'}")
        print(f"  Output: {output_path}")
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), is_color)
        
        if not out.isOpened():
            print(f"Error: Could not create video writer with codec '{codec}'")
            print("Trying alternative codec 'avc1'...")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), is_color)
            
            if not out.isOpened():
                print("Error: Could not create video writer with any codec")
                return False
        
        # Write frames
        for i, frame in enumerate(frames_list):
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Convert RGB to BGR (OpenCV uses BGR format)
            if is_color:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            out.write(frame_bgr)
            
            # Progress indicator
            if (i + 1) % 30 == 0 or (i + 1) == len(frames_list):
                print(f"  Writing frame {i + 1}/{len(frames_list)}...")
        
        # Release video writer
        out.release()
        
        print(f"\n✓ Video successfully created: {output_path}")
        print(f"  Duration: {len(frames_list) / fps:.2f} seconds")
        
        return True
    
    except Exception as e:
        print(f"Error reconstructing video: {e}")
        return False


def reconstruct_video_with_original_fps(frames_list, output_path, original_video_path):
    """
    Reconstruct a video using the FPS from the original video file.
    
    Parameters:
    -----------
    frames_list : list of numpy arrays
        List of frames to write
    output_path : str
        Path for output video
    original_video_path : str
        Path to original video (to extract FPS)
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Extract FPS from original video
        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open original video to get FPS. Using default 30 FPS.")
            fps = 30
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            print(f"Using FPS from original video: {fps:.2f}")
        
        # Reconstruct video
        return fps,len(frames_list),reconstruct_video(frames_list, output_path, fps=fps)
    
    except Exception as e:
        print(f"Error: {e}")
        return False
