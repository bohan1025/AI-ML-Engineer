import fiona
import Utils.Maubat_Utils.myGeoTools as mgt
import cv2


def getContoursCentroid(contours):
    """Compute centroid of contours."""

    # Initialize list to store centroid coordinates
    centroids = []

    # Iterate through contours
    for contour in contours:
        # Compute area of contour
        area = cv2.contourArea(contour)
        # Skip contour if area is zero
        if area == 0:
            continue
        # Compute moments of contour
        moments = cv2.moments(contour)
        # Compute centroid of contour
        px = int(moments["m10"] / moments["m00"])
        py = int(moments["m01"] / moments["m00"])
        # Add centroid coordinates to list
        centroids.append((px, py))
    return centroids


def exportShpFromCentroid(centroids, shpPath, gt):
    """Export a shapefile from a list of centroid, after
    remapping to coordinate system of the image.
    """
    # Save centroids as point shapefile
    schema = {"geometry": "Point", "properties": {}}
    with fiona.open(shpPath, "w", "ESRI Shapefile", schema) as output:
        for centroid in centroids:
            px, py = centroid
            cx, cy = mgt.pixel2coord(gt, px, py)
            point = {"type": "Point", "coordinates": (cx, cy)}
            output.write({"geometry": point, "properties": {}})
