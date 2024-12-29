import cv2
import tempfile
import os
from typing import Tuple

def compress_video(video_data: bytes, target_size_mb: int = 24) -> Tuple[bytes, bool]:
    """
    Compress video data to ensure it's below the target size while maintaining quality.
    
    Args:
        video_data: The original video data as bytes
        target_size_mb: Target size in MB (default 24 to stay safely under 25MB limit)
    
    Returns:
        Tuple containing:
        - The compressed video data as bytes
        - Boolean indicating if compression was successful
    """
    # Write input video to temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
        temp_input.write(video_data)
        temp_input_path = temp_input.name

    try:
        # Create temporary output file
        temp_output_path = temp_input_path + '_compressed.mp4'
        
        # Open the video
        cap = cv2.VideoCapture(temp_input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate target bitrate (80% of max allowed size to leave room for overhead)
        target_size_bytes = target_size_mb * 1024 * 1024 * 0.8
        duration = frame_count / fps
        target_bitrate = int((target_size_bytes * 8) / duration)

        # Initialize video writer with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        # Release everything
        cap.release()
        out.release()

        # Convert the output video to H.264 using ffmpeg for better compression
        final_output_path = temp_output_path + '_final.mp4'
        os.system(f'ffmpeg -i {temp_output_path} -c:v libx264 -preset medium -b:v {target_bitrate} {final_output_path}')

        # Read the compressed video
        with open(final_output_path, 'rb') as f:
            compressed_data = f.read()

        # Clean up temporary files
        os.unlink(temp_input_path)
        os.unlink(temp_output_path)
        os.unlink(final_output_path)

        # Verify size is within limit
        if len(compressed_data) <= (target_size_mb * 1024 * 1024):
            return compressed_data, True
        else:
            return video_data, False

    except Exception as e:
        # Clean up and return original data if compression fails
        try:
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
            os.unlink(final_output_path)
        except:
            pass
        return video_data, False