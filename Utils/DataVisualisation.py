import matplotlib.pyplot as plt

class DataVisualisation:
  def plot_history(self, history):
      loss = history.history['loss']
      val_loss = history.history['val_loss']
      acc = history.history['accuracy']
      val_acc = history.history['val_accuracy']

      plt.plot(loss)
      plt.plot(val_loss)
      plt.title('Model Loss')
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      plt.legend(['Train', 'Val'], loc='upper left')
      plt.show()

      plt.plot(acc)
      plt.plot(val_acc)
      plt.title('Model Accuracy')
      plt.ylabel('Accuracy')
      plt.xlabel('Epoch')
      plt.legend(['Train', 'Val'], loc='upper left')
      plt.show()
