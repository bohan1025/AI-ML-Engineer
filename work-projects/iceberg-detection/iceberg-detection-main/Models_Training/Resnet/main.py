import torch
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.models import resnet18
from osgeo import gdal, gdalconst, ogr
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
    precision_score,
)
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

from util import prepare_data, prepare_test_data, pointsAsPixels, plot_result_img
from dataset import CustomDataset
from train import fit, predict


def main():
    with open("config.json", "r", encoding="utf-8") as f:
        configs = json.load(f)
    imagefile_paths = configs[0]["imagefile_paths"]
    imagefile_paths = configs[0]["imagefile_paths_after_speckle_filter"]
    shapefile_paths = configs[0]["shapefile_paths"]

    pxl_range = configs[0]["pxl_range"]
    icb_no_icb_ratio = configs[0]["icb_no_icb_ratio"]
    data_with_testing = False
    # no_filter, mean_filter, gaussian_filter, scipy_median_filter, lee_filter
    # scipy_median_filter_with_enhance_contrast, scipy_median_filter_with_enhance_contrast_and_eroding_dilution, speckle_filter
    # lee_sigma_filter
    filter_name = "lee_sigma_filter"
    train_with_filter = True
    testing_only = False

    if not data_with_testing:
        test_imagefile_path = configs[0]["test_imagefile_path"]
        test_imagefile_path = (
            "sample/after_filter/S1A_EW_GRDM_1SDH_20230118T131123_84EF_S_1_Spk.tif"
        )
        # test_imagefile_path = "sample/all_images/S1A_EW_GRDM_1SDH_20230116T132728_A890_S_1.tif/S1A_EW_GRDM_1SDH_20230116T132728_A890_S_1.tif"
        # test_imagefile_path = "sample/all_images/S1A_EW_GRDM_1SDH_20230209T150539_E82B_S_1.tif/S1A_EW_GRDM_1SDH_20230209T150539_E82B_S_1.tif"
        # test_imagefile_path = "sample/all_images/S1A_EW_GRDM_1SDH_20230212T135159_17F5_S_1.tif/S1A_EW_GRDM_1SDH_20230212T135159_17F5_S_1.tif"
        # test_imagefile_path = "sample/all_images/S1A_EW_GRDM_1SDH_20230212T135259_06D2_S_1.tif/S1A_EW_GRDM_1SDH_20230212T135259_06D2_S_1.tif"
        test_shapefile_path = configs[0]["test_shapefile_path"]
        # test_shapefile_path = "sample/all_shapes/S1A_EW_GRDM_1SDH_20230116T132728_20230116T132828_046809_059CC4_A890/S1A_EW_GRDM_1SDH_20230116T132728_20230116T132828_046809_059CC4_A890.shp"
        # test_shapefile_path = "sample/all_shapes/S1A_EW_GRDM_1SDH_20230209T150539_20230209T150639_047160_05A886_E82B/S1A_EW_GRDM_1SDH_20230209T150539_20230209T150639_047160_05A886_E82B.shp"
        # test_shapefile_path = "sample/all_shapes/S1A_EW_GRDM_1SDH_20230212T135159_20230212T135259_047203_05A9F4_17F5/S1A_EW_GRDM_1SDH_20230212T135159_20230212T135259_047203_05A9F4_17F5.shp"
        # test_shapefile_path = "sample/all_shapes/S1A_EW_GRDM_1SDH_20230212T135259_20230212T135341_047203_05A9F4_06D2/S1A_EW_GRDM_1SDH_20230212T135259_20230212T135341_047203_05A9F4_06D2.shp"

    model_name = configs[0]["model_name"]
    # model_name = "efficientne
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Current Device: ", device)
    num_classes = 2
    epochs = 20
    batch_size = 32
    num_workers = 2
    learning_rates = 1e-4

    if train_with_filter:
        try:
            test_image_name = test_imagefile_path.split("/")[3][:-4]
        except:
            test_image_name = test_imagefile_path.split("/")[1][:-4]
        check_point_save_path = (
            f"check_points/{model_name}_best_model_{filter_name}_train_with_filter.pt"
        )
        figure_save_path = f"result_imgs/{model_name}_{filter_name}_train_with_filter_{test_image_name}.png"
        cutted_figure_save_path = f"result_imgs/{model_name}_{filter_name}_train_with_filter_cutted_{test_image_name}.png"
    else:
        try:
            test_image_name = test_imagefile_path.split("/")[3][:-4]
        except:
            test_image_name = test_imagefile_path.split("/")[1][:-4]
        check_point_save_path = (
            f"check_points/{model_name}_best_model_not_train_with_filter.pt"
        )
        figure_save_path = f"result_imgs/{model_name}_{filter_name}_not_train_with_filter_{test_image_name}.png"
        cutted_figure_save_path = f"result_imgs/{model_name}_{filter_name}_not_train_with_filter_cutted_{test_image_name}.png"

    if model_name == "resnet":
        model = resnet18(pretrained=True)
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
        model = model.to(device)
    elif model_name == "efficientnet":
        model = EfficientNet.from_pretrained("efficientnet-b3")
        model._fc = torch.nn.Linear(in_features=1536, out_features=2, bias=True)
        model = model.to(device)

    criterion = CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=learning_rates)

    if data_with_testing:
        (
            train_data,
            train_label,
            val_data,
            val_label,
            test_data,
            test_label,
        ) = prepare_data(
            imagefile_paths=imagefile_paths,
            shapefile_paths=shapefile_paths,
            pxl_range=pxl_range,
            icb_no_icb_ratio=icb_no_icb_ratio,
            with_testing=data_with_testing,
        )

    else:
        if not testing_only:
            if train_with_filter:
                train_data, train_label, val_data, val_label = prepare_data(
                    imagefile_paths=imagefile_paths,
                    shapefile_paths=shapefile_paths,
                    pxl_range=pxl_range,
                    icb_no_icb_ratio=icb_no_icb_ratio,
                    with_testing=data_with_testing,
                    filter_name=filter_name,
                )
            else:
                train_data, train_label, val_data, val_label = prepare_data(
                    imagefile_paths=imagefile_paths,
                    shapefile_paths=shapefile_paths,
                    pxl_range=pxl_range,
                    icb_no_icb_ratio=icb_no_icb_ratio,
                    with_testing=data_with_testing,
                )

        test_data, test_label, test_image_position = prepare_test_data(
            test_imagefile_path, test_shapefile_path, pxl_range, filter_name
        )

    if not testing_only:
        train_dataset = CustomDataset(train_data, train_label, model_name)
        val_dataset = CustomDataset(val_data, val_label, model_name)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    test_dataset = CustomDataset(test_data, test_label, model_name)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    if not testing_only:
        print("\nStart Training...")
        fit(
            model,
            epochs,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            check_point_save_path,
            device,
        )
        print("\nTraning Done.")

    print("\nStart Testing...")
    # load best checkpoint
    model = torch.load(check_point_save_path).to(device)
    test_outputs, test_labels = predict(model, test_loader, device)

    print("\nStart Cutted Testing...")
    new_test_outputs = []
    new_test_labels = []
    new_test_position = []
    for i in tqdm(range(len(test_image_position)), total=len(test_image_position)):
        if (
            test_image_position[i][0] >= 5000
            and test_image_position[i][1] <= 11000
            and test_image_position[i][2] >= 5000
            and test_image_position[i][3] <= 11000
        ):
            new_test_outputs.append(test_outputs[i])
            new_test_labels.append(test_labels[i])
            new_test_position.append(test_image_position[i])

    print("cutted f1 score: ", f1_score(new_test_labels, new_test_outputs))
    print(
        "cutted precision score: ", precision_score(new_test_labels, new_test_outputs)
    )
    print("cutted recall score: ", recall_score(new_test_labels, new_test_outputs))
    print("cutted confusion matrix: ")
    print(confusion_matrix(new_test_labels, new_test_outputs))
    print("\nTesting Done.")

    plot_result_img(
        test_imagefile_path,
        test_shapefile_path,
        test_outputs,
        test_image_position,
        figure_save_path,
        cutted_figure_save_path,
        pxl_range,
    )


if __name__ == "__main__":
    main()
