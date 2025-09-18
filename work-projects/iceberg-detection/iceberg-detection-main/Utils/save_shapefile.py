from Utils.exportShpFromContours import exportShpFromCentroid


def save_shapefile(results, save_path, gt):
    # Make into a shapefile and save output
    exportShpFromCentroid(results, save_path, gt)
