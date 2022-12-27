import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def preprocess_data(self, images, labels):
        images = images / 255.0
        labels = tf.keras.utils.to_categorical(labels)
        return train_test_split(images, labels, test_size=0.2)
