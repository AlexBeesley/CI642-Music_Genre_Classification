import os
from matplotlib import pyplot as plt
from DataManagement.DataLoader import DataLoader
from DataManagement.DataPreprocessor import DataPreprocessor
from Model.ModelCreator import ModelCreator
from Model.ModelManager import ModelManager
from Model.ModelTrainer import ModelTrainer
from Utils.ImagePredictor import ImagePredictor
from Utils.AudioToImage import convert_all_wav_to_png, convert_wav_to_png
from Utils.DataVisualisation import DataVisualisation

# INFO log level messages not printed, set to 0 to enable INFO logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

data_dir = 'C:/dev/CI642/Music_Genre_Classification/Data/genres_original_IMAGES'
if not os.path.exists(data_dir):
    print("No image data present. Generating image data from audio files...")
    convert_all_wav_to_png('C:/dev/CI642/Music_Genre_Classification/Data/genres_original')

model_file = 'model.h5'
epochs = 5

print("Initializing...")
data_loader = DataLoader(data_dir)
images, labels, class_names = data_loader.load_data()

print("Preprocessing data...")
data_preprocessor = DataPreprocessor()
X_train, X_val, Y_train, Y_val = data_preprocessor.preprocess_data(images, labels)

input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
num_classes = len(class_names)
print(f"Input shape: {input_shape}\nNumber of classes: {num_classes}")

print("Mounting...")
model_manager = ModelManager(model_file)
model = model_manager.load()

if not model:
    print("No model found, rebuilding:")
    model_creator = ModelCreator()

    print("\tCreating model...")
    model = model_creator.create_model(input_shape, num_classes)

    model.summary()

    print("\tTraining model...")
    model_trainer = ModelTrainer(model)
    history = model_trainer.train(X_train, Y_train, X_val, Y_val, epochs)

    visualisation = DataVisualisation()
    visualisation.plot_history(history)

    print("\tSaving model state...")
    model_manager.save(model)
else:
    print("Model found.")

print("Preparing prediction...")
image_predictor = ImagePredictor(model, class_names)
image = plt.imread(convert_wav_to_png('C:/dev/CI642/Music_Genre_Classification/Data/prediction_data/reggae.test.wav'))
image_predictor.predict(image)
