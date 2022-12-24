import os

import numpy as np
from matplotlib import pyplot as plt


class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self):
        images = []
        labels = []
        class_names = os.listdir(self.data_dir)
        class_names.sort()
        for class_name in class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                images.append(plt.imread(file_path))
                labels.append(class_names.index(class_name))
        images = np.array(images)
        labels = np.array(labels)
        return images, labels, class_names
