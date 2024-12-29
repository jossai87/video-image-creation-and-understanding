import boto3
from io import BytesIO
import json
import base64
from botocore.config import Config

#get a BytesIO object from file bytes
def get_bytesio_from_bytes(image_bytes):
    image_io = BytesIO(image_bytes)
    return image_io

#load the bytes from a file on disk
def get_bytes_from_file(file_path):
    with open(file_path, "rb") as image_file:
        file_bytes = image_file.read()
    return file_bytes


def get_response_from_model(prompt_content, image_bytes, model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime')
    
    try:
        if model_id == "amazon.titan-text-express-v1":
            # Format request for Titan Text Express model
            request = {
                "inputText": prompt_content,
                "textGenerationConfig": {
                    "maxTokenCount": 512,
                    "temperature": 0.5,
                }
            }
            
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(request)
            )
            
            response_body = json.loads(response["body"].read())
            output = response_body["results"][0]["outputText"]
            
        else:
            # Original code for other models
            image_message = {
                "role": "user",
                "content": [
                    {"text": "Image 1:"},
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {
                                "bytes": image_bytes
                            }
                        }
                    },
                    {"text": prompt_content + ' - Do not provide information on stuff that you cannot do.'}
                ],
            }
            
            response = bedrock.converse(
                modelId=model_id,
                messages=[image_message],
                inferenceConfig={
                    "maxTokens": 2000,
                    "temperature": 0
                },
            )
            
            output = response['output']['message']['content'][0]['text']
            
        return output
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise e
