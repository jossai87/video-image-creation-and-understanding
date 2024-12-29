import os
import tempfile
from moviepy.editor import VideoFileClip

def compress_video(video_data, max_size_mb=25):
    """
    Compress video data to target size.
    
    Args:
        video_data: Binary video data
        max_size_mb: Maximum size in MB for output video
        
    Returns:
        Compressed video data as bytes
    """
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
        temp_input.write(video_data)
        temp_input.flush()
        
        clip = VideoFileClip(temp_input.name)
        
        # Start with original quality
        quality = 0.9
        
        while True:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
                clip.write_videofile(temp_output.name, codec='libx264', audio_codec='aac', 
                                  ffmpeg_params=['-crf', str(int((1 - quality) * 51))],
                                  verbose=False, logger=None)
                
                # Check if file size is under max size
                if os.path.getsize(temp_output.name) <= max_size_mb * 1024 * 1024:
                    with open(temp_output.name, 'rb') as f:
                        compressed_data = f.read()
                    os.unlink(temp_output.name)
                    break
                    
                os.unlink(temp_output.name)
                quality -= 0.1
                
                if quality < 0.1:
                    raise ValueError("Could not compress video to target size")
        
        clip.close()
        os.unlink(temp_input.name)
        
    return compressed_data