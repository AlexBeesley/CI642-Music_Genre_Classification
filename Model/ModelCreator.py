import tensorflow as tf


class ModelCreator:
    def create_model(self, input_shape, num_classes):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        return model

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
