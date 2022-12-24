import numpy as np


class ImagePredictor:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

    def predict(self, target_image):
        target_image = target_image / 255.0
        target_image = np.expand_dims(target_image, axis=0)

        prediction = self.model.predict(target_image)

        class_index = np.argmax(prediction)
        class_probability = prediction[0, class_index]

        print('Predicted class:', self.class_names[class_index])
        print('Probability of each class:')
        for i in range(len(self.class_names)):
            print(self.class_names[i], ':', prediction[0, i])

        return class_probability
