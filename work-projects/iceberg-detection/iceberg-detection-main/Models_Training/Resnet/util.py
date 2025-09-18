import random

import fiona
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from osgeo import gdal, gdalconst, ogr
from pyproj import Transformer
from sklearn.model_selection import train_test_split
from scipy.ndimage import median_filter, gaussian_filter
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from skimage import img_as_ubyte

import Utils.Maubat_Utils.myGeoTools as mgt


def check_label(w, h, px, py, length):
    for x, y in zip(px, py):
        if x > w and x < w + length and y > h and y < h + length:
            return True
    return False


# Lee_filter with uniform filter
def lee_filter(image, window_size):
    print("Start lee filter")
    image = image.astype(float)
    # it makes the filter better for preprocessing the edge part
    padded_image = np.pad(image, pad_width=window_size // 2, mode="constant")
    # creating new filtered images and putting values inside
    filtered_image = np.zeros_like(image, dtype=float)

    for i in tqdm(range(image.shape[0]), total=image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i : i + window_size, j : j + window_size]
            center_pixel = window[window_size // 2, window_size // 2]
            # get the mean of all the window center
            window_mean = np.mean(window)
            window_variance = np.var(window)
            # Assuming constant mean of zero
            noise_variance = window_variance / 2
            # calculating the weight based on paper
            weight = (window_variance + 0.01) / (
                window_variance + noise_variance + 0.01
            )
            # calculating the de-speckle value of the new pixel
            filtered_pixel = center_pixel + weight * (image[i, j] - center_pixel)
            filtered_image[i, j] = filtered_pixel

    return filtered_image


def prepare_test_data(
    imagefile_path: str,
    shapefile_path: str,
    pxl_range: int = 10,
    filter_name: str = "no_filter",
    image=None,
):
    data = []
    label = []
    positions = []
    length = 2 * pxl_range

    if image is None:
        # Read the SAR data
        ds = gdal.Open(imagefile_path, gdalconst.GA_ReadOnly)
        gt = ds.GetGeoTransform()

        img = np.asarray(ds.GetRasterBand(1).ReadAsArray())
        img = img_as_ubyte(img)
        img[np.isnan(img)] = 0
        if filter_name == "mean_filter":
            img = cv2.blur(img, (5, 5))
        elif filter_name == "gaussian_filter":
            img = cv2.GaussianBlur(img, (3, 3), 0)
        elif filter_name == "scipy_median_filter":
            img = median_filter(img, size=9)
        elif filter_name == "scipy_median_filter_with_enhance_contrast":
            img = median_filter(img, size=9)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(9, 9))
            img = clahe.apply(img)
        elif (
            filter_name
            == "scipy_median_filter_with_enhance_contrast_and_eroding_dilution"
        ):
            img = median_filter(img, size=9)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(9, 9))
            img = clahe.apply(img)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            img = cv2.erode(img, kernel)
            img = cv2.dilate(img, kernel)
        elif filter_name == "lee_filter":
            img = lee_filter(img, 3)
    else:
        img = image

    if shapefile_path is not None:
        py, px = pointsAsPixels(shapefile_path, gt)
        height, width = img.shape

        print("Start to add test data...")
        for h in tqdm(range(0, height, length), total=int((height) / length)):
            for w in range(0, width, length):
                tmp_img = img[w : w + length, h : h + length]
                if tmp_img.shape != (2 * pxl_range, 2 * pxl_range):
                    continue
                data.append(np.array(tmp_img, dtype=np.float32))
                positions.append((w, w + length, h, h + length))
                if check_label(w, h, px, py, length):
                    label.append(1)
                else:
                    label.append(0)
    else:
        height, width = img.shape
        for h in tqdm(range(0, height, length), total=int((height) / length)):
            for w in range(0, width, length):
                tmp_img = img[w : w + length, h : h + length]
                if tmp_img.shape != (2 * pxl_range, 2 * pxl_range):
                    continue
                data.append(np.array(tmp_img, dtype=np.float32))
                positions.append((w, w + length, h, h + length))

    return data, label, positions


def prepare_data(
    imagefile_paths: list,
    shapefile_paths: list,
    pxl_range: int = 10,
    icb_no_icb_ratio: int = 2,
    with_testing: bool = False,
    filter_name: str = "no_filter",
):
    data = []
    label = []
    for i in range(len(imagefile_paths)):
        imagefile_path = imagefile_paths[i]
        shapefile_path = shapefile_paths[i]
        print("current: ", i + 1, " / ", len(imagefile_paths), " ", shapefile_path)

        # Read the SAR data
        ds = gdal.Open(imagefile_path, gdalconst.GA_ReadOnly)
        gt = ds.GetGeoTransform()

        img = np.asarray(ds.GetRasterBand(1).ReadAsArray())
        img[np.isnan(img)] = 0

        if filter_name == "mean_filter":
            img = cv2.blur(img, (5, 5))
        elif filter_name == "gaussian_filter":
            img = cv2.GaussianBlur(img, (3, 3), 0)
        elif filter_name == "scipy_median_filter":
            img = median_filter(img, size=9)
        elif filter_name == "scipy_median_filter_with_enhance_contrast":
            img = median_filter(img, size=9)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(9, 9))
            img = clahe.apply(img)
        elif (
            filter_name
            == "scipy_median_filter_with_enhance_contrast_and_eroding_dilution"
        ):
            img = median_filter(img, size=9)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(9, 9))
            img = clahe.apply(img)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            img = cv2.erode(img, kernel)
            img = cv2.dilate(img, kernel)
        elif filter_name == "lee_filter":
            img = lee_filter(img, 3)

        try:
            py, px = pointsAsPixels(shapefile_path, gt)
        except:
            print("error shape file: ", shapefile_path)
            continue
        height, width = img.shape

        print("Pos Samples")
        # add positive images and labels
        for i in tqdm(range(len(px)), total=len(px)):
            x = round(px[i])
            y = round(py[i])
            # no edges (all arrays must have same size)
            y_lbound = max(y - pxl_range, 0)
            y_ubound = min(y + pxl_range, width)
            x_lbound = max(x - pxl_range, 0)
            x_ubound = min(x + pxl_range, height)
            if y_lbound > 0 and y_ubound < width and x_lbound > 0 and x_ubound < height:
                pos_img = img[x_lbound:x_ubound, y_lbound:y_ubound]
                data.append(np.array(pos_img, dtype=np.float32))
                label.append(1)

        rxx = random.sample(range(0, width), icb_no_icb_ratio * len(px))
        rxy = random.sample(range(0, height), icb_no_icb_ratio * len(px))

        print("Neg Samples")
        for x, y in tqdm(zip(rxx, rxy), total=len(rxx)):
            # no overlap with icebergs
            if (
                min(
                    [
                        np.sqrt((x - round(px[i])) ** 2 + (y - round(py[i])) ** 2)
                        for i in range(len(px))
                    ]
                )
                > 2 * pxl_range
            ):
                # no edges (all arrays must have same size)
                y_lbound = max(y - pxl_range, 0)
                y_ubound = min(y + pxl_range, width)
                x_lbound = max(x - pxl_range, 0)
                x_ubound = min(x + pxl_range, height)
                if (
                    y_lbound > 0
                    and y_ubound < width
                    and x_lbound > 0
                    and x_ubound < height
                ):
                    neg_img = img[x_lbound:x_ubound, y_lbound:y_ubound]
                    # FOR VERIF, DELETE LATER
                    if neg_img.shape != (2 * pxl_range, 2 * pxl_range):
                        print(neg_img.shape)
                        print(y_lbound, y_ubound, x_lbound, x_ubound)
                    data.append(np.array(neg_img, dtype=np.float32))
                    label.append(0)

    if with_testing:
        train_val_data, test_data, train_val_label, test_label = train_test_split(
            data, label, test_size=0.2, random_state=42
        )
        train_data, val_data, train_label, val_label = train_test_split(
            train_val_data, train_val_label, test_size=0.1, random_state=42
        )
        return train_data, train_label, val_data, val_label, test_data, test_label
    else:
        train_data, val_data, train_label, val_label = train_test_split(
            data, label, test_size=0.2, random_state=42
        )
        return train_data, train_label, val_data, val_label


def pointsAsPixels(shapefile_path, gt):
    """Returns the coordinates of the shapefile in pixels,
    given the geotransform of the image. The shapefile must be in EPSG:4326,
    and the image in EPSG:3031
    """
    # Set the paths
    shapefile = shapefile_path

    # Open the shapefile
    label_data = ogr.Open(shapefile)
    layer = label_data.GetLayer()

    # Transform the shapefile coordinates to EPSG:3031
    transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
    x3031, y3031 = [], []
    for feat in tqdm(layer, total=len(layer)):
        geom = feat.GetGeometryRef()
        x, y = geom.GetX(), geom.GetY()  # in EPSG4326 coordinates
        xc, yc = transformer.transform(y, x)  # in EPSG3031 coordinates
        x3031.append(xc)
        y3031.append(yc)

    # Invert the GeoTransform matrix
    inv_gt = gdal.InvGeoTransform(gt)
    px = [
        inv_gt[0] + x3031[i] * inv_gt[1] + y3031[i] * inv_gt[2]
        for i in tqdm(range(len(x3031)), total=len(x3031))
    ]
    py = [
        inv_gt[3] + x3031[i] * inv_gt[4] + y3031[i] * inv_gt[5]
        for i in tqdm(range(len(y3031)), total=len(y3031))
    ]

    return px, py


def plot_result_img(
    test_imagefile_path: str,
    test_shapefile_path: str,
    predicted: list,
    predicted_position: list,
    figure_save_path: str,
    cutted_figure_save_path: str,
    pxl_range: int = 15,
):
    ds = gdal.Open(test_imagefile_path, gdalconst.GA_ReadOnly)
    gt = ds.GetGeoTransform()

    img = np.asarray(ds.GetRasterBand(1).ReadAsArray())
    py, px = pointsAsPixels(test_shapefile_path, gt)

    height, width = img.shape
    true_labels = np.zeros(img.shape, dtype=np.uint8)

    for i in tqdm(range(len(py)), total=len(py)):
        true_labels[
            max(round(px[i]) - pxl_range, 0) : min(round(px[i]) + pxl_range, height),
            max(round(py[i]) - pxl_range, 0) : min(round(py[i]) + pxl_range, width),
        ] = 1

    pred_labels = np.zeros(img.shape, dtype=np.uint8)

    for i in range(len(predicted)):
        w1 = predicted_position[i][0]
        w2 = predicted_position[i][1]
        h1 = predicted_position[i][2]
        h2 = predicted_position[i][3]
        pred_labels[w1:w2, h1:h2] = predicted[i]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(img)
    ax1.contour(true_labels)
    plt.title("true label")

    ax2 = fig.add_subplot(122)
    ax2.imshow(img)
    ax2.contour(pred_labels)
    plt.title("predicted")

    plt.savefig(figure_save_path)

    plt.cla()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(img[5000:11000, 5000:11000])
    ax1.contour(true_labels[5000:11000, 5000:11000])
    plt.title("true label")

    ax2 = fig.add_subplot(122)
    ax2.imshow(img[5000:11000, 5000:11000])
    ax2.contour(pred_labels[5000:11000, 5000:11000])
    plt.title("predicted")

    plt.savefig(cutted_figure_save_path)


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
