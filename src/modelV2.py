from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report
import os
import cv2

def image_preprocessing(image):
    sobel_ = cv2.Sobel(image,cv2.CV_64F, 0,1, ksize=5)
    return sobel_

def data_generator():
    '''
    Returns augmented, labeled data
    '''
    train_datagen = ImageDataGenerator(
        rescale= 1./255,
        preprocessor = image_preprocessing,
        validation_split= 0.1,
        shuffle = True
        )
    valid_datagen = ImageDataGenerator(rescale= 1./255)

    train_gen = train_datagen.flow_from_directory(model_params['train_dir'], target_size= (model_params['img_width'], model_params['img_height']), batch_size= model_params['batch_size'])
    valid_gen = valid_datagen.flow_from_directory(model_params['valid_dir'], target_size= (model_params['img_width'], model_params['img_height']), batch_size=model_params['batch_size'])
    
    return train_datagen, valid_datagen, train_gen, valid_gen

def search_and_destroy():
    folders = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','space','nothing']
    for alpha in folders:
        for file in os.listdir(f'../data/Train/{alpha}'):
            if file == '.ipynb_checkpoints':
                os.rmdir(file)
                print(f'File removed in {alpha}')         
        print(f'Nothing found in {alpha}')


def cnn_model():
    '''
    Returns Convolutional Neural Network model arcitecture 
    '''
    if K.image_data_format() == 'channels_first':
        input_shape = (3, model_params['img_width'], model_params['img_height'])
    else:
        input_shape = (model_params['img_width'], model_params['img_height'], 3)
    
    model = Sequential()

    model.add(Conv2D(64, (4,4), input_shape=input_shape, strides=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) 

    model.add(Conv2D(64, (4,4), activation = 'relu', strides=2))
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) 

    model.add(Conv2D(128, (4,4), activation='relu', strides=1))
     model.add(Activation('relu'))
    model.add(Dropout(0.5))     

    model.add(Conv2D(128, (4,4), strides=2))
     model.add(Activation('relu'))
    model.add(Dropout(0.5))  

    model.add(Conv2D(256, (4,4), strides=1))
     model.add(Activation('relu'))
    model.add(Dropout(0.5)) 

    model.add(Conv2D(256, (4,4), strides=2))
     model.add(Activation('relu'))
    model.add(Dropout(0.5)) 

    model.add(Flatten())
    model.add(Dense(500)) 
    model.add(Dropout(0.5)) 
    model.add(Dense(28), activation= 'softmax')
    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics=['accuracy']
    )

    return model

def model_evaluation_plot():
    '''
    Returns plot of model's performance as measured by accuracy metrics
    '''
    plt.plot(history.history['accuracy'], label= 'accuracy', color='gray')
    plt.plot(history.history['val_accuracy'], label= 'val_accuracy', color='orange', linestyle= '--')
    plt.plot(history.history['loss'], label= 'loss', color= 'gray')
    plt.plot(history.history['val_loss'], label= 'val_loss', color= 'orange', linestyle= '--')
    plt.xlabel('Epoch')
    plt.ylim([0,1])
    plt.title("Model Performance")
    plt.legend(loc='lower right')
    plt.show()

if __name__=='__main__':
    model_params = {
        'train_dir': '../data/Train',
        'valid_dir': '../data/Valid',
        'test_dir': '../data/Test',
        'batch_size': 64, # was 50
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
        steps_per_epoch = 100, # 58828 // model_params['batch_size'], # updates weight/backprop, can increase (more updates per epoch)
        epochs = model_params['epochs'],
        validation_data = valid_gen,
        validation_steps = 1,
        validation_split= 0.1, 
        callbacks= [tensorboard_callback, tensor_checkpoint]
    )

    # model.evaluate(valid_gen)
    model_evaluation_plot()
    # confusion_matrix_plot()
    model.save_weights('v3.h5')
