from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report
import numpy as np
from skimage.io import imread
import os
import PIL
import cv2
import seaborn as sns
sns.set()

def search_and_destroy():
    folders = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','space','nothing']
    for alpha in folders:
        for file in os.listdir(f'../../../test2/Training/{alpha}'):
            if file == '.ipynb_checkpoints':
                os.rmdir(file)
                print(f'File removed in {alpha}')         
        print(f'Nothing found in {alpha}')

def image_preprocessor(image):
    sobel_ = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return sobel_

def data_generator():
    '''
    Returns augmented, labeled data
    '''
    train_datagen = ImageDataGenerator(
        validation_split = 0.3,
        preprocessing_function = image_preprocessor,
        samplewise_center = True,
        samplewise_std_normalization = True,
        )
    valid_datagen = ImageDataGenerator(
        validation_split = 0.3,
        preprocessing_function = image_preprocessor,
        samplewise_center = True,
        samplewise_std_normalization = True,
    )

    train_gen = train_datagen.flow_from_directory(model_params['train_dir'], target_size= (model_params['img_width'], model_params['img_height']), batch_size= model_params['batch_size'], shuffle= True)
    valid_gen = valid_datagen.flow_from_directory(model_params['valid_dir'], target_size= (model_params['img_width'], model_params['img_height']), batch_size=model_params['batch_size'], shuffle= True)
    
    return train_datagen, valid_datagen, train_gen, valid_gen

def cnn_model():
    '''
    Returns Convolutional Neural Network model arcitecture 
    '''
    if K.image_data_format() == 'channels_first':
        input_shape = (3, model_params['img_width'], model_params['img_height'])
    else:
        input_shape = (model_params['img_width'], model_params['img_height'], 3)
    
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

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def model_evaluation_plot():
    '''
    Returns plot of model's performance as measured by accuracy metrics
    '''
    plt.plot(history.history['accuracy'], label= 'accuracy', color='gray')
    plt.plot(history.history['val_accuracy'], label= 'val_accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylim([0,1])
    plt.title("Model Performance")
    plt.legend(loc='lower right')
    plt.show()

# def confusion_matrix_plot():
#     '''
#     Returns confusion matrix of model's performance
#     '''
#     pred_raw = model.predict(valid_gen)
#     pred = np.argmax(pred_raw)
#     print('Confusion Matrix')
#     print(confusion_matrix(valid_gen.classes, pred))
#     print('Classification Report')
#     target_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','space','nothing']
#     print(classification_report(valid_gen.classes, pred, target_names=target_names))

def plot_confusion_matrix_with_default_options(y_pred, y_true, classes):
    cm = confusion_matrix(y_true, y_pred)
    with sns.axes_style('ticks'):
        plt.figure(figsize=(16, 16))
        plot_confusion_matrix(cm, classes)
        plt.show()
    return

if __name__=='__main__':
    model_params = {
        'train_dir': '../../../test2/Training',
        'valid_dir': '../../../test2/Validation',
        'batch_size': 64,
        'img_height': 64, 
        'img_width': 64,
        'epochs': 5
    }

    search_and_destroy()
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
    model = cnn_model()
    train_datagen, valid_datagen, train_gen, valid_gen = data_generator()
    model.summary()

    history = model.fit(
        train_gen,
        epochs = model_params['epochs'],
        validation_data = valid_gen,
        callbacks= [tensorboard_callback, tensor_checkpoint]
    )

    model_evaluation_plot()
    # confusion_matrix_plot()
    # model.save_weights('v3.h5')

