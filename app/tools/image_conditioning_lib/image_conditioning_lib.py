import boto3
import base64
import json
import logging
from PIL import Image
from io import BytesIO
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ImageError(Exception):
    "Custom exception for errors returned by models"
    def __init__(self, message):
        self.message = message

def get_target_size(model_id):
    """Determine target size based on model"""
    return (1280, 720) if "nova" in model_id.lower() else (512, 512)

def resize_image(image_bytes, model_id):
    """Resize image based on model requirements"""
    try:
        target_size = get_target_size(model_id)
        
        # Open image
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert back to bytes
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr.getvalue()
    except Exception as e:
        raise ImageError(f"Error resizing image: {str(e)}")

def encode_image_base64(image_bytes):
    """Convert image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')

def get_request_body(model_id, image_base64, prompt, negative_prompt=None, 
                    control_mode="CANNY_EDGE", control_strength=0.7):
    """Generate appropriate request body based on model"""
    target_size = get_target_size(model_id)
    
    base_config = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
            "conditionImage": image_base64,
            "controlMode": control_mode,
            "controlStrength": control_strength
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": target_size[1],
            "width": target_size[0],
            "cfgScale": 8.0
        }
    }
    
    # Add negative prompt if provided
    if negative_prompt:
        base_config["textToImageParams"]["negativeText"] = negative_prompt

    return json.dumps(base_config)

def generate_conditioned_image(model_id, image_bytes, prompt, 
                             negative_prompt=None, control_mode="CANNY_EDGE", 
                             control_strength=0.7):
    """
    Generate a conditioned image using the specified model.
    
    Args:
        model_id (str): The Bedrock model ID
        image_bytes (bytes): The input image bytes
        prompt (str): The text prompt
        negative_prompt (str, optional): Negative prompt
        control_mode (str): CANNY_EDGE or SEGMENTATION
        control_strength (float): Control strength (0.1 to 1.0)
    
    Returns:
        BytesIO: The generated image as a BytesIO object
    """
    try:
        # Initialize Bedrock client
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            config=Config(read_timeout=300)
        )

        # Resize image based on model requirements
        resized_image_bytes = resize_image(image_bytes, model_id)

        # Encode image to base64
        image_base64 = encode_image_base64(resized_image_bytes)

        # Get request body
        body = get_request_body(
            model_id=model_id,
            image_base64=image_base64,
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_mode=control_mode,
            control_strength=control_strength
        )

        # Invoke model
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )

        # Process response
        response_body = json.loads(response.get("body").read())

        # Check for errors
        if "error" in response_body and response_body["error"] is not None:
            raise ImageError(f"Image generation error: {response_body['error']}")

        # Get generated image
        base64_image = response_body.get("images")[0]
        image_bytes = base64.b64decode(base64_image)

        return BytesIO(image_bytes)

    except ClientError as err:
        error_message = err.response["Error"]["Message"]
        logger.error("Client error: %s", error_message)
        raise Exception(f"AWS Bedrock error: {error_message}")
    except Exception as e:
        logger.error("Error generating image: %s", str(e))
        raise
