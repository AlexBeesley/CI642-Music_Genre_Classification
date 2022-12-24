import os

from matplotlib import pyplot as plt

from DataManagement.DataLoader import DataLoader
from DataManagement.DataPreprocessor import DataPreprocessor
from Model.ModelCreator import ModelCreator
from Model.ModelTrainer import ModelTrainer
from Utils.ImagePredictor import ImagePredictor
from Utils.ModelManager import ModelManager

# INFO log level messages not printed, set to 0 to enable INFO logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

data_dir = 'C:/dev/CI642/Music_Genre_Classification/Data/images_original'
model_file = 'model.h5'
epochs = 5

print("Initializing...")
data_loader = DataLoader(data_dir)
images, labels, class_names = data_loader.load_data()

print("Preprocessing data...")
data_preprocessor = DataPreprocessor()
images, labels = data_preprocessor.preprocess_data(images, labels)

input_shape = (images.shape[1], images.shape[2], images.shape[3])
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

    print("\tTraining model...")
    model_trainer = ModelTrainer(model)
    model_trainer.train(images, labels, epochs)

    print("\tSaving model state...")
    model_manager.save(model)
else:
    print("Model found.")

print("Preparing prediction...")
image_predictor = ImagePredictor(model, class_names)
image = plt.imread('C:/dev/CI642/Music_Genre_Classification/Data/images_original/jazz/jazz00065.png')
image_predictor.predict(image)
