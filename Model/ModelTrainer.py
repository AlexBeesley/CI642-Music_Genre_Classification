import tensorflow as tf


class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, Y_train, X_val, Y_val, epochs, batch_size=16,
              early_stopping_patience=5, data_augmentation=False):

        learning_rate = 0.0001
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

        if data_augmentation:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2)
            datagen.fit(X_train)
            history = self.model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size), epochs=epochs,
                                     validation_data=(X_val, Y_val), callbacks=[early_stopping])
        else:
            history = self.model.fit(X_train, Y_train, batch_size=batch_size,
                                     epochs=epochs,
                                     validation_data=(X_val, Y_val),
                                     shuffle=False, callbacks=[early_stopping])
        return history

