import boto3
import json
import base64
from io import BytesIO
from random import randint
from botocore.config import Config

# Function to structure the request body based on the model type
import json
from random import randint

def get_image_generation_request_body(model_id, prompt, negative_prompt=None):
    rand_seed = randint(0, 2147483646)

    if "stability" in model_id.lower():
        # Handle Stability AI models (SD-XL, SD3, etc)
        if model_id == "stability.stable-diffusion-xl-v1":
            body = {
                "text_prompts": [
                    {
                        "text": prompt,
                        "weight": 1.0
                    }
                ],
                "cfg_scale": 10,
                "seed": rand_seed,
                "steps": 50
            }
            if negative_prompt:
                body["text_prompts"].append({
                    "text": negative_prompt,
                    "weight": -1.0
                })
        else:
            # For other Stability models like SD3, stable-image
            body = {
                "prompt": prompt
            }
            if negative_prompt:
                body["negative_prompt"] = negative_prompt
    else:
        # Handle Titan models
        body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "premium",
                "height": 768,
                "width": 1280,
                "cfgScale": 7.5,
                "seed": rand_seed,
            }
        }
        if negative_prompt:
            body['textToImageParams']['negativeText'] = negative_prompt

    return json.dumps(body)


def get_response_image(response, model_id=None):
    response_body = json.loads(response.get('body').read())
    
    if model_id == "amazon.nova-canvas-v1:0":
        try:
            # Get the base64 image directly from the response
            base64_image = response_body.get("images", [])[0]
            # Encode to ASCII then decode
            base64_bytes = base64_image.encode('ascii')
            image_data = base64.b64decode(base64_bytes)
            return BytesIO(image_data)
        except Exception as e:
            print(f"Error processing Nova response: {str(e)}")
            print(f"Response structure: {response_body}")
            raise ValueError("Failed to process Nova Canvas response")
    else:
        # Handle Titan and other models
        images = response_body.get('images')
        if images and len(images) > 0:
            image_data = base64.b64decode(images[0])
            return BytesIO(image_data)
            
    raise ValueError("No image data found in response")

def get_image_from_model(prompt_content, negative_prompt=None, model_id="amazon.titan-image-generator-v1"):
    # Select region based on model
    region = "us-west-2" if "stability" in model_id.lower() else "us-east-1"
    
    session = boto3.Session()
    bedrock = session.client(
        service_name='bedrock-runtime',
        region_name=region,
        config=Config(read_timeout=300)
    )
    
    try:
        body = get_image_generation_request_body(model_id, prompt_content, negative_prompt)
        
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            contentType="application/json",
            accept="application/json"
        )
        
        return get_response_image(response, model_id)
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise e



def get_image_from_model(prompt_content, negative_prompt=None, model_id="amazon.titan-image-generator-v1"):
    session = boto3.Session()
    
    # Check if model is from Stability AI to use us-west-2 region
    if any(x in model_id for x in ["stability", "Stability"]):
        bedrock = session.client(
            service_name='bedrock-runtime',
            region_name='us-west-2',
            config=Config(read_timeout=300)
        )
    else:
        bedrock = session.client(
            service_name='bedrock-runtime', 
            region_name='us-east-1',
            config=Config(read_timeout=300)
        )
    
    try:
        body = get_image_generation_request_body(model_id, prompt_content, negative_prompt)
        
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            contentType="application/json",
            accept="application/json"
        )
        
        return get_response_image(response, model_id)
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise e











# Function to handle the response and return the image as BytesIO
def get_response_image(response, model_id=None):
    response_body = response.get('body').read()
    print(f"Raw response for model {model_id}: {response_body}")  # Debug logging
    response = json.loads(response_body)
    
    # Handle Titan and other models
    images = response.get('images')
    if images and len(images) > 0:
        image_data = base64.b64decode(images[0])
        return BytesIO(image_data)
    raise ValueError("No image data found in response")

