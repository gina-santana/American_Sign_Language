from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import PIL
import cv2
import os
 
class CNNModel:
   classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S',
       'T','U','V','W','X','Y','Z','space','nothing']
 
   def __init__(self, train_dir, valid_dir, test_dir, model_type, batch_size = 64,
                img_height = 64, img_width = 64, epochs = 5, sobel_axis = 'x'):
       self.train_dir = train_dir
       self.valid_dir = valid_dir
       self.test_dir = test_dir
       self.batch_size = batch_size
       self.img_height = img_height
       self.img_width = img_width
       self.epochs = epochs
       self.sobel_axis = sobel_axis
       self.trained_model = None
       if model_type == 'simple':
           self.create_model_architecture = self._create_simple_model
       elif model_type == 'complex':
           self.create_model_architecture = self._create_complex_model
       elif model_type == 'transfer':
           self.create_model_architecture = self._create_transfer_model
       else:
           raise Exception('Model type of "simple", "complex", or "transfer" required.')
 
 
   def create_generator(self, image_dir, validation_split = 0.3, shuffle = True):
       '''
       Returns augmented, labeled data
       '''
       datagen = ImageDataGenerator(
           validation_split = validation_split,
           preprocessing_function = self._preprocessor,
           samplewise_center = True,
           samplewise_std_normalization = True,
       )
 
       return datagen.flow_from_directory(
           image_dir, target_size = (self.img_width, self.img_height),
           batch_size = self.batch_size, shuffle = shuffle
       )
 
 
   def create_model_evaluation_plot(self):
       '''
       Returns plot of model's performance as measured by accuracy metrics
       '''
       plt.plot(self.trained_model.history['accuracy'], label= 'accuracy', color='gray')
       plt.plot(self.trained_model.history['val_accuracy'], label= 'val_accuracy', color='orange')
       plt.xlabel('Epoch')
       plt.ylim([0,1])
       plt.title("Model Performance")
       plt.legend(loc='lower right')
       plt.show()       
 
 
   def evaluate_model(self, generator):
       '''
       Returns % loss and % accuracy
       '''
       evaluations = self.trained_model.evaluate(generator)
       for i in range(len(self.trained_model.metrics_names)):
           print("{}: {:.2f}%".format(
               self.trained_model.metrics_names[i], evaluations[i] * 100))
 
       predictions = self.trained_model.predict(generator)
 
       y_pred = np.argmax(predictions, axis=1)
       y_true = generator.classes
 
       return dict(y_pred=y_pred, y_true=y_true)
 
 
   def fit(self):
       model = self.create_model_architecture()
       model.summary()
       self.trained_model = model.fit(
           self.create_generator(self.train_dir),
           epochs = self.epochs,
           validation_data = self.create_generator(self.valid_dir),
           callbacks = self._get_callbacks()
       )
 
  
   def load_weights_from_file(self, model_weights_file):
       _file = Path(model_weights_file)
 
       if os.path.isfile(_file):
           model = self.create_model_architecture()
           model.load_weights(_file)
           self.trained_model = model
       else:
           'File does not exist.'
 
 
   def plot_confusion_matrix(self, y_pred, y_true, normalize = False,
                             title = 'Confusion Matrix', cmap = plt.cm.Oranges):
       cm = confusion_matrix(y_true, y_pred)
       with sns.axes_style('ticks'):
           plt.figure(figsize=(3, 3))
      
       if normalize:
           cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
           print('Normalized Confusion Matrix')
       else:
           print('Confusion Matrix - without normalization')
      
       plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
       plt.title(title)
       plt.colorbar()
       tick_marks = np.arange(len(self.classes))
       plt.xticks(tick_marks, self.classes, rotation = 45)
       plt.yticks(tick_marks, self.classes)
       fmt = '.2f' if normalize else 'd'
       threshold = cm.max() / 2.
      
       for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
           plt.text(j, i, format(cm[i, j], fmt), fontsize = 7,
                   horizontalalignment = 'center',
                   color = 'white' if cm[i, j] > threshold else 'black')
      
       plt.tight_layout()
       plt.ylabel('True Label')
       plt.xlabel('Predicted Label')
       plt.show()
       return
 
 
   def print_classification_report(self, evaluated_model):
       print(classification_report(**evaluated_model, target_names = self.classes))
 
 
   def save_model_to_file(self, _file):
       if self.trained_model != None:
           self.trained_model.save_weights(_file)
 
 
   def _create_complex_model(self):
       '''
       Returns Complex Convolutional Neural Network model architecture
       '''
       print('------------- Using Complex CNN Model ----------------')
 
       if K.image_data_format() == 'channels_first':
           input_shape = (3, self.img_width, self.img_height)
       else:
           input_shape = (self.img_width, self.img_height, 3)
      
       model = Sequential()
 
       model.add(Conv2D(64, kernel_size=4, strides=1, input_shape=input_shape))
       model.add(Activation('relu'))
 
       model.add(Conv2D(64, kernel_size=4, strides=2))
       model.add(Activation('relu'))
       model.add(Dropout(0.5))
      
       model.add(Conv2D(128, kernel_size=4, strides=1))
       model.add(Activation('relu'))
 
       model.add(Conv2D(128, kernel_size=4, strides=2))
       model.add(Activation('relu'))
       model.add(Dropout(0.5))
      
       model.add(Conv2D(256, kernel_size=4, strides=1))
       model.add(Activation('relu'))
 
       model.add(Conv2D(256, kernel_size=4, strides=2))
       model.add(Activation('relu'))
 
       model.add(Flatten())
       model.add(Dropout(0.5))
       model.add(Dense(512))
       model.add(Activation('relu'))
       model.add(Dense(28, activation='softmax'))
       model.compile(
           loss = 'categorical_crossentropy',
           optimizer = Adam(),
           metrics = ['accuracy']
       )
 
       return model
 
 
   def _create_simple_model(self):
       '''
       Returns Simple Convolutional Neural Network model architecture
       '''
       print('------------- Using Simple CNN Model ----------------')
  
       if K.image_data_format() == 'channels_first':
           input_shape = (3, self.img_width, self.img_height)
       else:
           input_shape = (self.img_width, self.img_height, 3)
      
       model = Sequential()
 
       model.add(Conv2D(64, (3,3), input_shape=input_shape))
       model.add(Activation('relu'))
       model.add(MaxPooling2D(pool_size= (2,2)))
       model.add(Dropout(0.25))
 
       model.add(Conv2D(16, (3,3)))
       model.add(Activation('relu'))
       model.add(BatchNormalization())   
       model.add(MaxPooling2D(pool_size= (2,2)))
       model.add(Dropout(0.3))
 
       model.add(Flatten())
       model.add(BatchNormalization())
       model.add(Dense(80))
       model.add(Activation('relu'))
       model.add(Dropout(0.5))
 
       model.add(BatchNormalization())
       model.add(Dense(28))
       model.add(Activation('softmax'))
       model.compile(
           loss = 'categorical_crossentropy',
           optimizer = Adam(learning_rate = 0.001),
           metrics = ['accuracy']
       )
 
       return model
 
 
   def _create_transfer_model(self, weights = 'imagenet', trainable_index = 128):
       '''
       Returns Transfer Convolutional Neural Network model architecture
       '''
       print('------------- Using Transfer Learning CNN Model ----------------')
 
       base_model = Xception(weights = weights,
                             include_top = False,
                             input_shape = (self.img_width, self.img_height, 3))
          
       model = base_model.output
       model = GlobalAveragePooling2D()(model)
       predictions = Dense(len(self.classes), activation = 'softmax')(model)
       model = Model(inputs = base_model.input, outputs = predictions)
 
       for layer in model.layers[:trainable_index]:
           layer.trainable = False
       for layer in model.layers[trainable_index:]:
           layer.trainable = True
      
       return model
 
 
   def _get_callbacks(self):
       tensorboard_callback = TensorBoard(
           log_dir="../data/logs",
           histogram_freq=0,
           write_graph=True,
           write_images=False,
           update_freq= 'epoch',
           profile_batch= 2,
           embeddings_freq= 0,
           embeddings_metadata= None
       )
 
       tensor_checkpoint = ModelCheckpoint('../data/models', save_best_only= True)
 
       return [tensorboard_callback, tensor_checkpoint]
 
   def _preprocessor(self, image):
       if self.sobel_axis == 'x':
           return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
       else:
           return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

