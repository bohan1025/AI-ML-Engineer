import cv2
import numpy as np
from Utils.exportShpFromContours import getContoursCentroid


def contouring(image):
    """Makes and saves a shapefile from the contours of the image.
    Args:
        image (str): Image to be contoured.
        gt (str): The geotransform of the image.
        save_name (str): The name to save the shapefile as.

    Returns: str: A message indicating the success of the operation."""

    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the original image
    coutour_orgn = np.copy(image)
    coutour_orgn = cv2.drawContours(coutour_orgn, contours, -1, (255, 0, 0), 2)
    centroids = getContoursCentroid(contours)

    return centroids
