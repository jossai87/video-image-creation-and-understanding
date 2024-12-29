import boto3
import json
import base64
import logging
from PIL import Image
from io import BytesIO
from random import randint
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ImageError(Exception):
    "Custom exception for errors returned by models"
    def __init__(self, message):
        self.message = message

# Utility functions remain the same
def get_bytesio_from_bytes(image_bytes):
    return BytesIO(image_bytes)

def get_png_base64(image):
    png_io = BytesIO()
    image.save(png_io, format="PNG")
    return base64.b64encode(png_io.getvalue()).decode("utf-8")

def get_image_from_bytes(image_bytes):
    return Image.open(BytesIO(image_bytes))

def get_bytes_from_file(file_path):
    with open(file_path, "rb") as image_file:
        return image_file.read()

def get_mask_image_base64(target_width, target_height, position, inside_width, inside_height):
    inside_color_value = (0, 0, 0)
    outside_color_value = (255, 255, 255)
    
    mask_image = Image.new("RGB", (target_width, target_height), outside_color_value)
    original_image_shape = Image.new("RGB", (inside_width, inside_height), inside_color_value)
    mask_image.paste(original_image_shape, position)
    return get_png_base64(mask_image)

def get_request_body(model_id, prompt, input_image_bytes, negative_prompt=None, 
                    vertical_alignment=0.5, horizontal_alignment=0.5, 
                    outpainting_mode="DEFAULT", quality="standard"):
    """Generate request body based on selected model"""
    
    original_image = get_image_from_bytes(input_image_bytes)
    original_width, original_height = original_image.size
    target_width = 1024
    target_height = 1024
    
    position = (
        int((target_width - original_width) * horizontal_alignment), 
        int((target_height - original_height) * vertical_alignment),
    )
    
    extended_image = Image.new("RGB", (target_width, target_height), (235, 235, 235))
    extended_image.paste(original_image, position)
    input_image_base64 = get_png_base64(extended_image)
    mask_image_base64 = get_mask_image_base64(target_width, target_height, position, original_width, original_height)

    if model_id == "amazon.nova-canvas-v1:0":
        body = {
            "taskType": "OUTPAINTING",
            "outPaintingParams": {
                "text": prompt,
                "image": input_image_base64,
                "maskImage": mask_image_base64,
                "outPaintingMode": outpainting_mode
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": target_height,
                "width": target_width,
                "cfgScale": 8.0
            }
        }
        if negative_prompt:
            body["outPaintingParams"]["negativeText"] = negative_prompt
    else:  # Titan model
        body = {
            "taskType": "OUTPAINTING",
            "outPaintingParams": {
                "image": input_image_base64,
                "maskImage": mask_image_base64,
                "text": prompt,
                "outPaintingMode": outpainting_mode
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": quality,
                "width": target_width,
                "height": target_height,
                "cfgScale": 8.0,
                "seed": randint(0, 100000)
            }
        }
        if negative_prompt:
            body["outPaintingParams"]["negativeText"] = negative_prompt

    return json.dumps(body)

def get_image_from_model(prompt_content, image_bytes, negative_prompt=None, 
                        vertical_alignment=0.5, horizontal_alignment=0.5, 
                        model_id="amazon.titan-image-generator-v2:0",
                        outpainting_mode="DEFAULT", quality="standard"):
    """Main function to generate image using selected model"""
    try:
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            config=Config(read_timeout=300)
        )
        
        body = get_request_body(
            model_id,
            prompt_content, 
            image_bytes, 
            negative_prompt=negative_prompt, 
            vertical_alignment=vertical_alignment, 
            horizontal_alignment=horizontal_alignment,
            outpainting_mode=outpainting_mode,
            quality=quality
        )
        
        response = bedrock.invoke_model(
            body=body, 
            modelId=model_id, 
            contentType="application/json", 
            accept="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        
        if "error" in response_body and response_body["error"] is not None:
            raise ImageError(f"Image generation error: {response_body['error']}")
            
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
