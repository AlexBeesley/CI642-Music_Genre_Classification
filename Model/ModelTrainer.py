class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, images, labels, epochs):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(images, labels, epochs=epochs)
