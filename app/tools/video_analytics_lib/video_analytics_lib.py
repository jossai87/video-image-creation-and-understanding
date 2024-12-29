
import base64
import boto3
import json
import time
import tempfile
import subprocess
import os

def upload_file_to_s3(file_content, bucket_name, file_name, content_type):
    """Upload a file to S3"""
    try:
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=file_content,
            ContentType=content_type
        )
        return f"Successfully uploaded {file_name} to S3"
    except Exception as e:
        return f"Error uploading file to S3: {str(e)}"


def compress_video(video_data, max_size_mb=25, chunk_duration=30):
    """
    Compress or chunk video data if it exceeds size limit.
    """
    try:
        current_size_mb = len(video_data) / (1024 * 1024)
        if current_size_mb <= max_size_mb:
            return video_data, True

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as input_file:
            input_file.write(video_data)
            input_file.flush()
            
            # Get video duration
            duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                          "-of", "default=noprint_wrappers=1:nokey=1", input_file.name]
            duration = float(subprocess.check_output(duration_cmd).decode().strip())
            
            # If video is too long, take first chunk
            if duration > chunk_duration:
                output_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                # Extract first chunk_duration seconds
                chunk_cmd = [
                    "ffmpeg", "-i", input_file.name,
                    "-t", str(chunk_duration),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-y",
                    output_file.name
                ]
                subprocess.run(chunk_cmd, check=True, capture_output=True)
                
                # Read chunked video
                with open(output_file.name, 'rb') as f:
                    chunked_data = f.read()
                
                # Clean up
                os.unlink(input_file.name)
                os.unlink(output_file.name)
                
                # Verify chunk size
                chunk_size_mb = len(chunked_data) / (1024 * 1024)
                if chunk_size_mb <= max_size_mb:
                    return chunked_data, True
            
            os.unlink(input_file.name)
            return None, False
            
    except Exception as e:
        print(f"Error in video processing: {str(e)}")
        return None, False


def analyze_video_with_nova(bucket_name, video_filename, prompt="Analyze and describe this video content", model_id="us.amazon.nova-lite-v1:0"):
    """
    Analyze video using Amazon Bedrock with Nova model.
    """
    try:
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        s3_client = boto3.client('s3')
        video_object = s3_client.get_object(Bucket=bucket_name, Key=video_filename)
        binary_data = video_object['Body'].read()
        
        # Compress video if needed
        compressed_data, compression_success = compress_video(binary_data)
        
        # Check compression result
        if not compression_success:
            return {
                'status': 'error',
                'message': 'Could not compress video to required size limit of 25MB'
            }
            
        # Add notification if video was chunked
        if len(compressed_data) != len(binary_data):
            print("Note: Video was chunked to first 30 seconds due to size limits")
            
        base64_string = base64.b64encode(compressed_data).decode("utf-8")

        system_list = [{
            "text": "You are an expert media analyst. When the user provides you with a video, provide an analysis or description."
        }]

        message_list = [{
            "role": "user",
            "content": [
                {
                    "video": {
                        "format": "mp4",
                        "source": {"bytes": base64_string}
                    }
                },
                {
                    "text": prompt
                }
            ]
        }]

        inf_params = {
            "max_new_tokens": 300,
            "top_p": 0.1,
            "top_k": 20,
            "temperature": 0.3
        }

        native_request = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "system": system_list,
            "inferenceConfig": inf_params
        }

        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(native_request)
        )
        model_response = json.loads(response["body"].read())
        content_text = model_response["output"]["message"]["content"][0]["text"]

        return {
            'status': 'success',
            'analysis': content_text
        }

    except boto3.exceptions.Boto3Error as e:
        return {
            'status': 'error',
            'message': f"Boto3 error: {str(e)}"
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


def get_video_url_from_s3(bucket_name, video_filename):
    """Generate a presigned URL for accessing a video in S3"""
    try:
        s3_client = boto3.client('s3')
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': video_filename
            },
            ExpiresIn=3600
        )
        return url
    except Exception as e:
        return str(e)

def generate_audio_description(text, voice_id, bucket_name, output_filename):
    """
    Generate audio from text using Amazon Polly and store in S3.
    
    Args:
        text (str): The text to convert to speech
        voice_id (str): The Polly voice ID to use
        bucket_name (str): S3 bucket to store the audio
        output_filename (str): Filename for the generated audio
    
    Returns:
        dict: Result containing status, audio URL, and audio data
    """
    try:
        # Initialize Polly client
        polly_client = boto3.client('polly')
        
        # Request speech synthesis
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId=voice_id,
            Engine='neural'
        )
        
        # Get audio stream
        audio_stream = response['AudioStream'].read()
        
        # Upload to S3
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=bucket_name,
            Key=output_filename,
            Body=audio_stream,
            ContentType='audio/mpeg'
        )
        
        # Generate presigned URL
        audio_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': output_filename
            },
            ExpiresIn=3600
        )
        
        return {
            'status': 'success',
            'audio_url': audio_url,
            'audio_data': audio_stream
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

def generate_audio(bucket_name, text, voice_id):
    """
    Generate audio from text using Amazon Polly.
    
    Args:
        text (str): Text to convert to speech
        voice_id (str): Polly voice ID to use
    
    Returns:
        dict: Result containing status and audio URL
    """
    try:
        polly_client = boto3.client('polly')
        
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId=voice_id,
            Engine='neural'
        )
        
        if "AudioStream" in response:
            # Generate a unique filename
            audio_file = f"analysis_audio_{int(time.time())}.mp3"
            
            # Upload to S3
            s3_client = boto3.client('s3')
            s3_client.put_object(
                Bucket=bucket_name,
                Key=audio_file,
                Body=response['AudioStream'].read(),
                ContentType='audio/mpeg'
            )
            
            # Generate presigned URL
            audio_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': bucket_name,
                    'Key': audio_file
                },
                ExpiresIn=3600
            )
            
            return {
                'status': 'success',
                'audio_url': audio_url
            }
        else:
            return {
                'status': 'error',
                'message': 'No audio stream in response'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

