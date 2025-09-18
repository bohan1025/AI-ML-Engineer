from tqdm import tqdm
from Models_Training.Resnet.train import predict
from Models_Training.Resnet.dataset import CustomDataset
from Models_Training.Resnet.util import prepare_test_data
import torch
from torch.utils.data import DataLoader
from osgeo import gdal, gdalconst


class ResnetPredictor:
    """
    A class used to prepare an image and make predictions using a ResNet model.

    Attributes:
    model_path (str): The path to the ResNet model.
    batch_size (int): The batch size to use when creating the DataLoader.
    device (str): The device to use for computations ('cpu' or 'cuda').
    model (torch.nn.Module): The loaded ResNet model.
    """

    def __init__(self, model_path, batch_size=32, device=None):
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(self.model_path, map_location=torch.device("cpu")).to(
            self.device
        )

    def prepare_image(
        self,
        image_path,
        test_shapefile_path=None,
        pxl_range=15,
        filter_name="no_filter",
        image=None,
    ):
        """
        Prepares an image for prediction by splitting it into patches and creating a DataLoader.

        Parameters:
        image_path (str): The path to the image to prepare.
        test_shapefile_path (str): The path to the shapefile for the test data. Defaults to None.
        pxl_range (int): The pixel range to use when splitting the image into patches. Defaults to 15.
        filter_name (str): The name of the filter to use when preparing the image. Defaults to "no_filter".
        """
        ds = gdal.Open(image_path, gdalconst.GA_ReadOnly)
        self.gt = ds.GetGeoTransform()
        test_data, _, self.test_image_position = prepare_test_data(
            image_path, test_shapefile_path, pxl_range, filter_name, image
        )
        test_label = [0] * len(test_data)
        test_dataset = CustomDataset(test_data, test_label, "resnet")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def predict(self):
        self.test_outputs, _ = predict(self.model, self.test_loader, self.device)

    def get_centroids(self, image_path, image=None):
        self.prepare_image(image_path, image=image)
        self.predict()

        all_position = []
        for i in tqdm(
            range(len(self.test_image_position)), total=len(self.test_image_position)
        ):
            if self.test_outputs[i] == 1:
                all_position.append(self.test_image_position[i])

        return [((pos[3] + pos[2]) / 2, (pos[1] + pos[0]) / 2) for pos in all_position]
