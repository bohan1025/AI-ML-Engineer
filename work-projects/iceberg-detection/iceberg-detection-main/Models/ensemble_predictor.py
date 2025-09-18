import numpy as np
import time
from Utils.oneToOneMatchSequential import oneToOneMatchSequential
from Utils.getPoints import pointsAsPixels4326


class EnsemblePredictor:
    def __init__(self, gt, max_dist=50):
        self.gt = gt
        self.max_dist = max_dist
        self.votes = {}

    def process(self, shapefiles_paths):
        # Store coordinates of each shapefile in the form:
        # [array[cfar_x, cfar_y], array[cnn_x, cnn_y], array[resn_x, resn_y]...]
        all_coords = [
            np.array(pointsAsPixels4326(path, self.gt)) for path in shapefiles_paths
        ]

        # Match shapefiles
        votes = oneToOneMatchSequential(all_coords, self.max_dist)

        # Get the number of votes for each point
        max_votes = max(votes.values())
        votes_nb = [
            list(filter(lambda x: x[1] == i, votes.items()))
            for i in range(1, max_votes + 1)
        ]

        # Get the coordinates of each point, grouped by number of votes
        votes_coord = [np.array([x[0] for x in votes_i]) for votes_i in votes_nb]

        self.votes = votes
        self.votes_coord = votes_coord

    def save_votes(self, path):
        np.save(path, self.votes)

    def get_votes(self, shapefiles_paths):
        self.process(shapefiles_paths)

        return self.votes, self.votes_coord
