import numpy as np
import cv2

bbox_color_dict = {}
def draw_bbox(frame: np.ndarray, bbox: np.ndarray, id: int, cls: str):
    """Draw the bounding box on a for a single detected object."""

    x1, y1, x2, y2 = bbox

    # get random colour based on class
    if cls not in bbox_color_dict:
        bbox_color_dict[cls] = np.random.randint(0, 255, size=(3)).tolist()
    bbox_color = (86,0,255) #bbox_color_dict[cls]
    
    text = f"{id}: {cls}"
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 3)
    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), bbox_color, -1)
    cv2.putText(frame, text, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, [255, 255, 255], 2)
    
def resize_with_aspect_ratio(image, target_size):

    target_width = target_size[0]
    target_height = target_size[1]
    # Get original image dimensions
    (h, w) = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = w / h

    # Determine the new width and height
    if target_width / target_height > aspect_ratio:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    else:
        new_height = int(target_width / aspect_ratio)
        new_width = target_width

    # Resize the image to the new dimensions
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a new image with the target size (640x480) and fill with a black background
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate the center to place the resized image
    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2

    # Place the resized image in the center of the target image
    padded_image[top:top+new_height, left:left+new_width] = resized_image

    return padded_image


# def add_hud_to_video(frame, roll, battery_lvl, cur_obj, altitude):
    
    
    