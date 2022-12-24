import tensorflow as tf

class DataPreprocessor:
    def preprocess_data(self, images, labels):
        images = images / 255.0
        labels = tf.keras.utils.to_categorical(labels)
        return images, labels
