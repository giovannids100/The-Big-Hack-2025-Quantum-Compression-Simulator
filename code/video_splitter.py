import cv2
import numpy as np
from PIL import Image

def load_video_frames(video_path, size=(64, 64), max_frames=None, frame_skip=1):
    """
    Load video frames, resize them, and convert to numpy arrays.
    
    Parameters:
    -----------
    video_path : str
        Path to the MP4 video file
    size : tuple
        Target size for frames (width, height)
    max_frames : int, optional
        Maximum number of frames to extract (None = all frames)
    frame_skip : int
        Extract every N-th frame (1 = all frames, 2 = every other frame, etc.)
    
    Returns:
    --------
    list of numpy arrays
        List containing all extracted and resized frames
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video loaded: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip frames according to frame_skip parameter
            if frame_count % frame_skip == 0:
                # Convert BGR (OpenCV format) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                frame_resized = cv2.resize(frame_rgb, size, interpolation=cv2.INTER_AREA)
                
                # Convert to numpy array
                frames.append(frame_resized)
                extracted_count += 1
                
                # Check if we've reached max_frames limit
                if max_frames and extracted_count >= max_frames:
                    print(f"Reached maximum frame limit: {max_frames}")
                    break
            
            frame_count += 1
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}/{total_frames}...")
        
        cap.release()
        
        print(f"Successfully extracted {len(frames)} frames from video")
        return frames
    
    except Exception as e:
        print(f"Error loading video: {e}")
        return None