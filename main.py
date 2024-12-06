import json
from PIL import Image
import requests
import io
import base64
from typing import Dict,Any
from rembg import remove
import boto3
from datetime import datetime, timezone
import os

s3_client = boto3.client('s3')
S3_BUCKET_S3_URL = "s3://kifferai-static-assets-prod"
S3_BUCKET_URL = "https://kifferai-static-assets-prod.s3.ap-south-1.amazonaws.com"
CDN_URL = "https://static-assets.kifferai.com"
AWS_REGION = os.environ.get('AWS_REGION')

def get_public_url(url):
    if url.startswith(S3_BUCKET_URL):
        return url.replace(S3_BUCKET_URL, CDN_URL)
    elif url.startswith(S3_BUCKET_S3_URL):
        return url.replace(S3_BUCKET_S3_URL, CDN_URL)
    return url 


def upload_base64_to_s3(bucket, key, image_base64, region):
    """
    Uploads a base64 image to an S3 bucket.

    Parameters:
        bucket (str): Name of the S3 bucket.
        key (str): The desired file name (key) in the S3 bucket.
        image_base64 (str): The base64-encoded image data.
    """
    # Decode the base64 image
    image_data = base64.b64decode(image_base64)
    
    # Upload the image to S3
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=image_data,
            ContentType="image/png"
        )
        print(f"Uploaded {key} to S3 bucket {bucket}.")

        https_url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

        public_url = get_public_url(https_url)
        return public_url

    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise


def handler(event, context):
    """
    AWS Lambda handler function to process image data and return masks, mask centers, and a base64 cutout.
    
    Parameters:
        event (Dict[str, Any]): The input event data from API Gateway.
        context (Any): The runtime information provided by Lambda.
    
    Returns:
        Dict[str, Any]: A dictionary containing the masks, mask centers, and base64-encoded cutout image.
    """
    # Retrieve query parameters
    url = event.get('queryStringParameters', {}).get('url', None)
    orgId = event.get('queryStringParameters', {}).get("orgId", "defaultOrg")
    requestId = event.get('queryStringParameters', {}).get("requestId", "defaultRequest")

    # Load image from URL
    if url:
        response = requests.get(url)
        image_data = response.content
    else:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No URL provided"})
        }

    # Convert image data to a PIL image
    img = Image.open(io.BytesIO(image_data))

    # Call the remove function
    result = remove(img)

    mask_base64_list = []
    bounding_rects = []

    for mask in result["masks"]:
        buffer = io.BytesIO()
        mask.save(buffer, format="PNG")  # Save each mask image to a buffer
        mask_base64_list.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))  # Convert to base64

        # Calculate bounding box
        bbox = mask.getbbox()
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            bounding_rect = {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            }
            bounding_rects.append(bounding_rect)

        else:
            bounding_rects.append(None)

    mask_centers_converted = [
            {"x": int(center["x"]) if center["x"] is not None else None,
            "y": int(center["y"]) if center["y"] is not None else None}
            for center in result["mask_centers"] 
        ]
    
    if bounding_rects and bounding_rects[0]:
        bbox = bounding_rects[0]
        x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]

        # Decode the base64 image
        image_data = base64.b64decode(result["image_base64"])
        imageNew = Image.open(io.BytesIO(image_data))

        # Crop the image using the first bounding box
        cropped_image = imageNew.crop((x_min, y_min, x_max, y_max))

        # Save the cropped image to a buffer
        buffer = io.BytesIO()
        cropped_image.save(buffer, format="PNG")
        cropped_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        cropped_base64 = None

    # Generate S3 keys
    iso_timestamp = datetime.now(timezone.utc).isoformat().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    base_key_path = f"{orgId}/{requestId}/cutoutImages/"
    s3_image_key = f"{base_key_path}image_{iso_timestamp}.png"
    cropped_s3_image_key = f"{base_key_path}cropped_image_{iso_timestamp}.png"

    full_image_url = upload_base64_to_s3(
        bucket="kifferai-static-assets-prod",
        key=s3_image_key,
        image_base64=result["image_base64"],
        region=AWS_REGION
    )

    cropped_image_url = None
    if cropped_base64:
        cropped_image_url = upload_base64_to_s3(
            bucket="kifferai-static-assets-prod",
            key=cropped_s3_image_key,
            image_base64=cropped_base64,
            region=AWS_REGION
        )



    response_data: Dict[str, Any] = {
        "bounding_rects": bounding_rects,
        "cropped_image_url": cropped_image_url,
        "image_url": full_image_url,
    }

    # Return the results as a JSON response
    return {
        "statusCode": 200,
        "body": response_data,  # Convert the result dictionary to JSON
        "headers": {
            "Content-Type": "application/json"
        }
    }

