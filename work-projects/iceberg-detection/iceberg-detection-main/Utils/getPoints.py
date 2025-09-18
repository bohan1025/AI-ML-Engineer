from osgeo import gdal, ogr
from pyproj import Transformer


def pointsAsPixels(shapefile_path, gt):
    """Returns the coordinates of the shapefile in pixels,
    given the geotransform of the image. The shapefile must be in EPSG:4326,
    and the image in EPSG:3031"""
    # Set the paths
    shapefile = shapefile_path

    # Open the shapefile
    label_data = ogr.Open(shapefile)
    layer = label_data.GetLayer()

    # Transform the shapefile coordinates to EPSG:3031
    transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
    x3031, y3031 = [], []

    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        x, y = geom.GetX(), geom.GetY()  # in EPSG4326 coordinates
        xc, yc = transformer.transform(y, x)  # in EPSG3031 coordinates
        x3031.append(xc)
        y3031.append(yc)

    # Invert the GeoTransform matrix
    inv_gt = gdal.InvGeoTransform(gt)
    px = [
        inv_gt[0] + x3031[i] * inv_gt[1] + y3031[i] * inv_gt[2]
        for i in range(len(x3031))
    ]
    py = [
        inv_gt[3] + x3031[i] * inv_gt[4] + y3031[i] * inv_gt[5]
        for i in range(len(y3031))
    ]

    return px, py


def pointsAsPixels4326(shapefile_path, gt):
    """Returns the coordinates of the shapefile in pixels,
    given the geotransform of the image. The shapefile and the image must be in EPSG:4326
    """
    # Set the paths
    shapefile = shapefile_path

    # Open the shapefile
    label_data = ogr.Open(shapefile)
    layer = label_data.GetLayer()

    # Transform the shapefile coordinates to EPSG:3031
    x4326, y4326 = [], []

    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        x, y = geom.GetX(), geom.GetY()  # in EPSG4326 coordinates
        x4326.append(x)
        y4326.append(y)

    # Invert the GeoTransform matrix
    inv_gt = gdal.InvGeoTransform(gt)
    px = [
        inv_gt[0] + x4326[i] * inv_gt[1] + y4326[i] * inv_gt[2]
        for i in range(len(x4326))
    ]
    py = [
        inv_gt[3] + y4326[i] * inv_gt[4] + y4326[i] * inv_gt[5]
        for i in range(len(x4326))
    ]

    return px, py


def pointsAs3031(shapefile_path):
    """Returns the coordinates of the shapefile in EPSG:3031"""
    # Set the paths
    shapefile = shapefile_path

    # Open the shapefile
    label_data = ogr.Open(shapefile)
    layer = label_data.GetLayer()

    # Transform the shapefile coordinates to EPSG:3031
    transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
    x3031, y3031 = [], []
    for feat in layer:
        geom = feat.GetGeometryRef()
        x, y = geom.GetX(), geom.GetY()  # in EPSG4326 coordinates
        xc, yc = transformer.transform(y, x)  # in EPSG3031 coordinates
        x3031.append(xc)
        y3031.append(yc)
    return x3031, y3031


def pointsAs4326(shapefile_path):
    """Returns the coordinates of the shapefile in EPSG:4326"""
    # Set the paths
    shapefile = shapefile_path

    # Open the shapefile
    label_data = ogr.Open(shapefile)
    layer = label_data.GetLayer()

    # Coordinates in EPSG:4326 by default
    x4326, y4326 = [], []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        x, y = geom.GetX(), geom.GetY()  # in EPSG4326 coordinates
        x4326.append(x)
        y4326.append(y)
    return x4326, y4326
