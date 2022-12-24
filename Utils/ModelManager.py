import os
import tensorflow as tf


class ModelManager:
    def __init__(self, model_file):
        self.model_file = model_file

    def save(self, model):
        model.save(self.model_file)

    def load(self):
        if os.path.exists(self.model_file):
            model = tf.keras.models.load_model(self.model_file)
            return model
        else:
            return None
