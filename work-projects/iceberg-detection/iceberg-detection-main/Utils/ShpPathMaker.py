import os


class ShpPathMaker:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def make_shp_path(self, image_name, save_name):
        """Make a shapefile path from a shapefile name. All files saved
        using this PathMaker will be saved in the folder specified in
        the constructor.
        """
        out_folder = f"{self.folder_path}/{image_name}/{save_name}"
        os.makedirs(out_folder, exist_ok=True)
        out_path = f"{out_folder}/{save_name}.shp"
        return out_path
