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

def generate_image_variation(
    input_image_base64,
    prompt_text,
    negative_text="Nothing",
    similarity_strength=0.7,
    width=512,
    height=512,
    cfg_scale=8.0,
    model_id="amazon.nova-canvas-v1:0"
):
    """
    Generate an image variation using selected AWS Bedrock model.
    """
    try:
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            config=Config(read_timeout=300)
        )

        # Handle empty negative prompt
        negative_text = negative_text if negative_text else ""

        # Prepare the request body based on model
        if model_id == "amazon.nova-canvas-v1:0":
            body = json.dumps({
                "taskType": "IMAGE_VARIATION",
                "imageVariationParams": {
                    "text": prompt_text,
                    "negativeText": negative_text,
                    "images": [input_image_base64],
                    "similarityStrength": similarity_strength,
                },
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "height": height,
                    "width": width,
                    "cfgScale": cfg_scale
                }
            })
        else:  # Titan Image Generator V2
            body = json.dumps({
                "taskType": "IMAGE_VARIATION",
                "imageVariationParams": {
                    "text": prompt_text,
                    "negativeText": negative_text,
                    "images": [input_image_base64],
                    "similarityStrength": similarity_strength,
                },
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "height": height,
                    "width": width,
                    "cfgScale": cfg_scale
                }
            })

        # Rest of the function remains the same
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        
        if "error" in response_body and response_body["error"] is not None:
            raise Exception(f"Image generation error: {response_body['error']}")
        
        base64_image = response_body.get("images")[0]
        image_bytes = base64.b64decode(base64_image)
        
        return Image.open(io.BytesIO(image_bytes))

    except ClientError as err:
        error_message = err.response["Error"]["Message"]
        logger.error("Client error: %s", error_message)
        raise Exception(f"AWS Bedrock error: {error_message}")
    except Exception as e:
        logger.error("Error generating image variation: %s", str(e))
        raise

