"""
Construction Document Cropper Proof of Concept
Usage: construction_doc_cropper --image 'path_to_image'
"""
import cv2
import numpy as np
import argparse

def find_largest_object_crop(img_path, thresh=210):
    """
    Assumption: the largest object in a construction doc will be the floor plan containing the doors we want. 
    
    Find the largest object in an image using connected component analysis
    and save images related to cropping and a cropped version around that object.
    
    This method is just a proof of concept around using connected component analysis for getting the
    floor plan out of a larger construction document without training an autocropping model. 
    
    Args:
        image_path (str): Path to the input image
        thresh (int): Threshold for binary conversion (0-255)
    """
    
    # Read the image
    original_image = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to create binary image
    # Note: some research into best thresh method is needed
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite("./results/crop_experiments/binary.jpg", binary)
    
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Find the largest component (excluding background which is label 0)
    largest_component_label = 1
    largest_area = 0
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_component_label = i
    
    if largest_area == 0:
        raise ValueError("No objects found in the image")
    
    # Get bounding box of the largest component
    x = stats[largest_component_label, cv2.CC_STAT_LEFT]
    y = stats[largest_component_label, cv2.CC_STAT_TOP]
    w = stats[largest_component_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_component_label, cv2.CC_STAT_HEIGHT]
    
    # Add some padding around the object (optional)
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(original_image.shape[1] - x, w + 2 * padding)
    h = min(original_image.shape[0] - y, h + 2 * padding)
    
    # Crop the original image around the largest object
    cropped_image = original_image[y:y+h, x:x+w]
    
    cv2.imwrite("./results/crop_experiments/cropped_image.jpg", cropped_image)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--thresh', type=int, help='thresh value for cca', default=210)

    args = parser.parse_args()

    find_largest_object_crop(img_path=args.image, thresh=args.thresh)