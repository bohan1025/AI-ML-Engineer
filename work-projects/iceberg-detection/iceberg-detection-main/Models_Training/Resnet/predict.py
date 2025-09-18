import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
    precision_score,
)
from osgeo import gdal, gdalconst, ogr

from train import predict
from dataset import CustomDataset
from util import (
    prepare_data,
    prepare_test_data,
    pointsAsPixels,
    getContoursCentroid,
    exportShpFromCentroid,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
check_point_save_path = "check_points/best_model.pt"
# test_imagefile_path = "sample/after_filter/S1A_EW_GRDM_1SDH_20230116T132728_A890_S_1_Spk.tif"
test_imagefile_path = "sample/all_images/S1A_EW_GRDM_1SDH_20230116T132728_A890_S_1.tif/S1A_EW_GRDM_1SDH_20230116T132728_A890_S_1.tif"
# test_shapefile_path = "sample/all_shapes/S1A_EW_GRDM_1SDH_20230116T132728_20230116T132828_046809_059CC4_A890/S1A_EW_GRDM_1SDH_20230116T132728_20230116T132828_046809_059CC4_A890.shp"
test_shapefile_path = None
pxl_range = 15
model_name = "resnet"
batch_size = 32
num_workers = 2

# load best checkpoint
model = torch.load(check_point_save_path).to(device)

filter_name = "no_filter"
ds = gdal.Open(test_imagefile_path, gdalconst.GA_ReadOnly)
gt = ds.GetGeoTransform()
test_data, test_label, test_image_position = prepare_test_data(
    test_imagefile_path, test_shapefile_path, pxl_range, filter_name
)
test_label = [0] * len(test_data)
test_dataset = CustomDataset(test_data, test_label, model_name)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

test_outputs, test_labels = predict(model, test_loader, device)

all_position = []
for i in tqdm(range(len(test_image_position)), total=len(test_image_position)):
    if test_outputs[i] == 1:
        all_position.append(test_image_position[i])

centroids = []
for pos in all_position:
    centroids.append(((pos[3] + pos[2]) / 2, (pos[1] + pos[0]) / 2))
# centroids = getContoursCentroid(all_position)
exportShpFromCentroid(centroids, "A890_resnet_mean_filter.shp", gt)
