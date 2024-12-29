import boto3
import base64
from botocore.config import Config
from PIL import Image
from io import BytesIO


def get_presigned_url(bucket_name, object_key, expiration=3600):
    """Generate a presigned URL for accessing a video in S3"""
    try:
        s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': object_key
            },
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        print(f"Error generating presigned URL: {str(e)}")
        return None



def render_image_to_video_tab(text_prompt, image_bytes=None, duration=6, fps=24, dimension="1280x720", output_bucket="my-nova-videos"):
    """Generate video using Amazon Bedrock with Nova Reel model"""
    try:
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        
        # Base model input
        model_input = {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": {
                "text": text_prompt
            },
            "videoGenerationConfig": {
                "durationSeconds": duration,
                "fps": fps,
                "dimension": dimension,
                "seed": 0
            }
        }
        
        # Add image input if provided
        if image_bytes:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            model_input["textToVideoParams"]["images"] = [{
                "format": "png",
                "source": {
                    "bytes": base64_image
                }
            }]

        # Start async video generation
        response = client.start_async_invoke(
            modelId="amazon.nova-reel-v1:0",
            modelInput=model_input,
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": f"s3://{output_bucket}"
                }
            }
        )
        
        return {
            'status': 'success',
            'invocationArn': response.get('invocationArn'),
            'response': response
        }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


def check_job_status(invocation_arn):
    """Check the status of an async video generation job"""
    try:
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        response = client.get_async_invoke(invocationArn=invocation_arn)
        return response.get('status')
    except Exception as e:
        return f"Error checking status: {str(e)}"

def get_video_url(bucket_name, invocation_arn):
    """Get the video URL using presigned URL"""
    try:
        # Extract the job ID from the invocation ARN
        job_id = invocation_arn.split('/')[-1]
        # Construct the S3 object key
        object_key = f"{job_id}/output.mp4"
        # Generate presigned URL
        presigned_url = get_presigned_url(bucket_name, object_key)
        return presigned_url
    except Exception as e:
        print(f"Error getting video URL: {str(e)}")
        return None

def resize_image_to_1280x720(image_bytes):
    """Resize image to 1280x720 while maintaining aspect ratio"""
    try:
        # Open the image
        img = Image.open(BytesIO(image_bytes))
        
        # Calculate aspect ratio
        target_ratio = 1280 / 720
        current_ratio = img.width / img.height
        
        # Determine new dimensions
        if current_ratio > target_ratio:
            # Image is too wide
            new_width = int(720 * target_ratio)
            new_height = 720
        else:
            # Image is too tall
            new_width = 1280
            new_height = int(1280 / target_ratio)
            
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create white background
        final_img = Image.new('RGB', (1280, 720), 'white')
        
        # Calculate position to paste resized image
        paste_x = (1280 - new_width) // 2
        paste_y = (720 - new_height) // 2
        
        # Paste resized image onto white background
        final_img.paste(resized_img, (paste_x, paste_y))
        
        # Convert back to bytes
        img_byte_arr = BytesIO()
        final_img.save(img_byte_arr, format=img.format if img.format else 'PNG')
        img_byte_arr.seek(0)
        
        return {
            'status': 'success',
            'image_bytes': img_byte_arr.getvalue(),
            'original_size': f"{img.width}x{img.height}",
            'was_resized': (img.width, img.height) != (1280, 720)
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }
