class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, Y_train, X_val, Y_val, epochs):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(X_train, Y_train,
                                 epochs=epochs, batch_size=16,
                                 validation_data=(X_val, Y_val),
                                 shuffle=False)
        return history
