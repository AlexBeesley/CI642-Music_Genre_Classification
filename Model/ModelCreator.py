import tensorflow as tf


class ModelCreator:
    def create_model(self, input_shape, num_classes):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.04)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        return model
