import base64
import io
import json
import logging
import boto3
from PIL import Image
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ImageError(Exception):
    def __init__(self, message):
        self.message = message

def get_target_size(model_id):
    """Determine target size based on model"""
    return (1280, 720) if "nova" in model_id.lower() else (512, 512)

def resize_image_if_needed(image_bytes, model_id):
    """Resize image based on model requirements"""
    try:
        target_size = get_target_size(model_id)
        
        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert back to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    
    except Exception as e:
        raise ImageError(f"Error resizing image: {str(e)}")

def get_request_body(model_id, prompt, colors, negative_prompt=None, 
                    reference_image=None, width=1024, height=1024,
                    num_images=1, cfg_scale=8.0):
    """Generate appropriate request body based on model"""
    
    # Ensure colors have '#' prefix
    formatted_colors = []
    for color in colors:
        if not color.startswith('#'):
            formatted_colors.append(f"#{color}")
        else:
            formatted_colors.append(color)
    
    params = {
        "text": prompt,
        "colors": formatted_colors  # Use the properly formatted colors
    }
    
    if negative_prompt:
        params["negativeText"] = negative_prompt
        
    if reference_image:
        image_bytes = resize_image_if_needed(reference_image.getvalue(), model_id)
        params["referenceImage"] = base64.b64encode(image_bytes).decode('utf-8')

    body = {
        "taskType": "COLOR_GUIDED_GENERATION",
        "colorGuidedGenerationParams": params,
        "imageGenerationConfig": {
            "numberOfImages": num_images,
            "height": height,
            "width": width,
            "cfgScale": cfg_scale,
            "seed": 0
        }
    }
    
    return json.dumps(body)


def generate_images(model_id, prompt, colors, negative_prompt=None,
                   reference_image=None, width=1024, height=1024,
                   num_images=1, cfg_scale=8.0):
    """Generate images using color-guided generation"""
    try:
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            config=Config(read_timeout=300)
        )

        body = get_request_body(
            model_id, prompt, colors, negative_prompt,
            reference_image, width, height, num_images, cfg_scale
        )

        # Log the request body for debugging
        logger.info(f"Request body: {body}")

        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get('body').read())
        
        # Log the response for debugging
        logger.info(f"Response body: {response_body}")
        
        if "error" in response_body and response_body["error"] is not None:
            raise ImageError(f"Image generation error: {response_body['error']}")

        # Handle different response formats
        images_list = []
        if isinstance(response_body.get("images"), list):
            images_list = response_body["images"]
        elif isinstance(response_body.get("image"), str):
            images_list = [response_body["image"]]
        else:
            logger.warning(f"Unexpected response format: {response_body}")
            
        # Generate multiple images if needed
        generated_images = []
        for base64_image in images_list:
            image_bytes = base64.b64decode(base64_image)
            generated_images.append(Image.open(io.BytesIO(image_bytes)))
            
        # If we got fewer images than requested, log a warning
        if len(generated_images) < num_images:
            logger.warning(f"Requested {num_images} images but received {len(generated_images)}")

        return generated_images

    except ClientError as err:
        error_message = err.response["Error"]["Message"]
        logger.error(f"Client error: {error_message}")
        raise Exception(f"AWS Bedrock error: {error_message}")
    except Exception as e:
        logger.error(f"Error in color-guided generation: {str(e)}")
        raise
