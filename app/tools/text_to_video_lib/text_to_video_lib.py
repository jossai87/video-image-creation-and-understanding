import boto3
import base64
import json

def render_text_to_video_tab(prompt_text, image_bytes=None, duration=6, fps=24, dimension="1280x720", seed=0, generation_type="Text to Video", output_bucket="demo-portal-videos-jossai-east1"):
    try:
        # Initialize the boto3 client
        client = boto3.client("bedrock-runtime", region_name="us-east-1")

        # Prepare the model input
        model_input = {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": {
                "text": prompt_text
            },
            "videoGenerationConfig": {
                "durationSeconds": duration,
                "fps": fps,
                "dimension": dimension,
                "seed": seed
            }
        }

        # Add image input if provided
        if generation_type == "Image to Video" and image_bytes:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            model_input["imageInput"] = {
                "image": base64_image
            }

        # Start async video generation with proper S3 path
        response = client.start_async_invoke(
            modelId="amazon.nova-reel-v1:0",
            modelInput=model_input,
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": f"s3://{output_bucket}"
                }
            }
        )
        
        if 'invocationArn' not in response:
            raise ValueError("No invocation ARN in response")
            
        return response['invocationArn']
    except Exception as e:
        print(f"Error generating video: {str(e)}")
        raise e

def get_presigned_url(bucket_name, object_key, expiration=3600):
    """Generate a presigned URL for accessing a video in S3"""
    try:
        s3_client = boto3.client('s3')
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
