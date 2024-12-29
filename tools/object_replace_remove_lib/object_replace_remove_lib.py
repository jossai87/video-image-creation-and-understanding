import boto3
import json
import base64
from io import BytesIO
from random import randint
from botocore.config import Config

def get_bytesio_from_bytes(image_bytes):
    return BytesIO(image_bytes)

def get_base64_from_bytes(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

def get_bytes_from_file(file_path):
    with open(file_path, "rb") as image_file:
        return image_file.read()

def get_titan_request_body(prompt, image_bytes, mask_prompt=None, negative_prompt=None):
    input_image_base64 = get_base64_from_bytes(image_bytes)
    
    body = {
        "taskType": "INPAINTING",
        "inPaintingParams": {
            "image": input_image_base64,
            "maskPrompt": mask_prompt if mask_prompt else "empty area in the image",
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "premium",
            "height": 512,
            "width": 512,
            "cfgScale": 8.0,
            "seed": randint(0, 100000),
        },
    }
    
    if prompt:
        body['inPaintingParams']['text'] = prompt
    if negative_prompt:
        body['inPaintingParams']['negativeText'] = negative_prompt
    
    return json.dumps(body)

def get_nova_canvas_request_body(prompt, image_bytes, mask_prompt=None, negative_prompt=None):
    input_image_base64 = get_base64_from_bytes(image_bytes)
    
    body = {
        "taskType": "INPAINTING",
        "inPaintingParams": {
            "image": input_image_base64,
            "maskPrompt": mask_prompt if mask_prompt else "empty area in the image",
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 512,
            "width": 512,
            "cfgScale": 8.0
        }
    }
    
    if prompt:
        body['inPaintingParams']['text'] = prompt
    if negative_prompt:
        body['inPaintingParams']['negativeText'] = negative_prompt
        
    return json.dumps(body)

def get_response_image(response, model_id):
    response_body = json.loads(response.get('body').read())
    
    if 'nova-canvas' in model_id:
        image_data = base64.b64decode(response_body.get('images')[0])
    else:
        image_data = base64.b64decode(response_body.get('images')[0])
        
    return BytesIO(image_data)

def get_image_from_model(prompt_content, image_bytes, mask_prompt=None, negative_prompt=None, model_id="amazon.titan-image-generator-v2:0"):
    session = boto3.Session()
    bedrock = session.client(
        service_name='bedrock-runtime',
        config=Config(read_timeout=300)
    )
    
    try:
        if 'nova-canvas' in model_id:
            body = get_nova_canvas_request_body(prompt_content, image_bytes, mask_prompt, negative_prompt)
        else:
            body = get_titan_request_body(prompt_content, image_bytes, mask_prompt, negative_prompt)
        
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            contentType="application/json",
            accept="application/json"
        )
        
        return get_response_image(response, model_id)
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None
