import numpy as np
from torch.utils.data import DataLoader


class CNNPredictor:
    """
    A class used to process images for a Convolutional Neural Network (CNN) model.

    Attributes:
    model (nn.Module): The CNN model to use for predictions.
    batch_size (int): The batch size to use when creating the DataLoader.
    window_size (int): The size of the window to use when splitting the image into patches.
    step (int): The step size to use when splitting the image into patches.

    Methods:
    split_image(image): Splits the given image into patches of size window_size x window_size,
                        stepping over the image in increments of step size.
    create_dataloader(patches): Creates a DataLoader from the given patches.
    predict(dataloader, image): Makes predictions on the patches in the given DataLoader using the model,
                                and returns an output array of the same size as the image.
    process_image(image): Processes the given image using the model, and returns an output array of the same size as the image.
    """

    def __init__(self, model, batch_size=64, window_size=30, step=60):
        self.model = model
        self.batch_size = batch_size
        self.window_size = window_size
        self.step = step

    def split_image(self, image):
        """
        Splits the given image into patches of size window_size x window_size, stepping over the image in increments of step size.

        Parameters:
        image (np.array): The image to split into patches.

        Returns:
        inp (list): A list of tuples, where each tuple contains a patch and its top-left coordinates.
        """
        ysize, xsize = image.shape
        inp = []
        for x in range(0, image.shape[1] - self.window_size // 2, self.step):
            for y in range(0, image.shape[0] - self.window_size // 2, self.step):
                x_max = min(x + self.window_size // 2, xsize)
                x_min = max(x - self.window_size // 2, 0)
                y_max = min(y + self.window_size // 2, ysize)
                y_min = max(y - self.window_size // 2, 0)
                sub = image[y_min:y_max, x_min:x_max]
                if sub.shape == (self.window_size, self.window_size):
                    inp.append((sub, (y, x)))
        return inp

    def create_dataloader(self, patches):
        return DataLoader(patches, batch_size=self.batch_size, shuffle=False)

    def predict(self, dataloader, image):
        output = np.zeros_like(image)
        print("output shape:", output.shape)
        for images, (y, x) in dataloader:
            try:
                images = images.float()
                images = images.unsqueeze(1)
                outputs = self.model(images)
                pre = (outputs > 0.5).float()
                for i in range(len(pre)):
                    if pre[i] == 1:
                        output[y, x] = 1
            except IndexError as e:
                print("IndexError:", e)
                print("y, x:", y, x)
                pass
        return output

    def process_image(self, image):
        patches = self.split_image(image)
        dataloader = self.create_dataloader(patches)
        output = self.predict(dataloader, image)
        return output
