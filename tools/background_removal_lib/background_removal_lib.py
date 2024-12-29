import boto3
import json
import base64
from io import BytesIO
from botocore.config import Config

def remove_background(image_bytes):
    """
    Remove background from image using Nova Canvas model.
    
    Args:
        image_bytes (bytes): Input image bytes
    Returns:
        BytesIO: Processed image without background
    """
    try:
        # Initialize Bedrock client
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',
            config=Config(read_timeout=300)
        )

        # Encode image as base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare request body
        body = json.dumps({
            "taskType": "BACKGROUND_REMOVAL",
            "backgroundRemovalParams": {
                "image": base64_image
            }
        })

        # Call Nova Canvas model
        response = bedrock.invoke_model(
            body=body,
            modelId="amazon.nova-canvas-v1:0",
            contentType="application/json",
            accept="application/json"
        )

        # Process response
        response_body = json.loads(response.get("body").read())
        
        # Check for errors
        if "error" in response_body:
            raise Exception(f"Model error: {response_body['error']}")

        # Get generated image
        base64_result = response_body.get("images")[0]
        image_data = base64.b64decode(base64_result)
        
        return BytesIO(image_data)

    except Exception as e:
        print(f"Error removing background: {str(e)}")
        raise e
