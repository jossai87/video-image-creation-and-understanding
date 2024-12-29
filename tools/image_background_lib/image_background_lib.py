import boto3
import json
import base64
from io import BytesIO
from random import randint
from botocore.config import Config

# Helper functions
def get_bytesio_from_bytes(image_bytes):
    return BytesIO(image_bytes)

def get_base64_from_bytes(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

def get_bytes_from_file(file_path):
    with open(file_path, "rb") as image_file:
        return image_file.read()




# Function to create request body for Titan models
def get_titan_image_background_replacement_request_body(prompt, image_bytes, mask_prompt, negative_prompt=None, outpainting_mode="DEFAULT", number_of_images=1):
    input_image_base64 = get_base64_from_bytes(image_bytes)
    body = {
        "taskType": "OUTPAINTING",
        "outPaintingParams": {
            "image": input_image_base64,
            "text": prompt,
            "maskPrompt": mask_prompt,
            "outPaintingMode": outpainting_mode,
        },
        "imageGenerationConfig": {
            "numberOfImages": number_of_images,
            "quality": "premium",
            "height": 512,
            "width": 512,
            "cfgScale": 8.0,
            "seed": randint(0, 100000),
        },
    }
    if negative_prompt:
        body['outPaintingParams']['negativeText'] = negative_prompt
    return json.dumps(body)


# Function to process Titan model responses
def get_titan_response_images(response):
    response_body = json.loads(response['body'].read())
    images = response_body.get('images', [])
    return [BytesIO(base64.b64decode(image_data)) for image_data in images]

def get_nova_canvas_request_body(prompt, image_bytes, mask_prompt, negative_prompt=None, outpainting_mode="DEFAULT", number_of_images=1):
    input_image_base64 = get_base64_from_bytes(image_bytes)
    body = {
        "taskType": "OUTPAINTING",
        "outPaintingParams": {
            "text": prompt,
            "image": input_image_base64,
            "outPaintingMode": outpainting_mode
        },
        "imageGenerationConfig": {
            "numberOfImages": number_of_images,
            "height": 512,
            "width": 512,
            "cfgScale": 8.0
        }
    }
    
    if mask_prompt:
        body["outPaintingParams"]["maskPrompt"] = mask_prompt
    if negative_prompt:
        body["outPaintingParams"]["negativeText"] = negative_prompt
        
    return json.dumps(body)

def get_nova_canvas_response_images(response):
    response_body = json.loads(response['body'].read())
    images = response_body.get('images', [])  # Nova Canvas returns images directly
    if not images:
        return None
    return BytesIO(base64.b64decode(images[0]))  # Return first image

def get_image_from_model(prompt_content, image_bytes, model_id, mask_prompt=None, negative_prompt=None, outpainting_mode="DEFAULT", number_of_images=1):
    session = boto3.Session()
    bedrock = session.client(
        service_name='bedrock-runtime',
        config=Config(read_timeout=300)  # Added timeout for longer operations
    )

    try:
        if "nova-canvas" in model_id:
            body = get_nova_canvas_request_body(
                prompt_content,
                image_bytes,
                mask_prompt=mask_prompt,
                negative_prompt=negative_prompt,
                outpainting_mode=outpainting_mode,
                number_of_images=number_of_images
            )
            response = bedrock.invoke_model(
                body=body,
                modelId=model_id,
                contentType="application/json",
                accept="application/json"
            )
            return get_nova_canvas_response_images(response)
        else:
            body = get_titan_image_background_replacement_request_body(
                prompt_content,
                image_bytes,
                mask_prompt=mask_prompt,
                negative_prompt=negative_prompt,
                outpainting_mode=outpainting_mode,
                number_of_images=number_of_images
            )
            response = bedrock.invoke_model(
                body=body,
                modelId=model_id,
                contentType="application/json",
                accept="application/json"
            )
            return get_titan_response_images(response)[0]  # Return first image for consistency
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

