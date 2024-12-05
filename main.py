import json
from PIL import Image
import requests
import io
import base64
from typing import Dict,Any
from rembg import remove  # Ensure this import is correct based on your project structure



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

    response_data: Dict[str, Any] = {
        "masks": mask_base64_list,
        "mask_centers": mask_centers_converted,
        "bounding_rects": bounding_rects,
        "cropped_image_base64": cropped_base64,
        "image_base64": result["image_base64"],
    }

    # Return the results as a JSON response
    return {
        "statusCode": 200,
        "body": json.dumps(response_data),  # Convert the result dictionary to JSON
        "headers": {
            "Content-Type": "application/json"
        }
    }

