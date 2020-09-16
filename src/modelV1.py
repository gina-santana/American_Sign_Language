from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.metrics import plot_confusion_matrix
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
import os
import PIL

# def img_to_array():
#     stacked_arrays = np.empty([2105, 120000, 28])
#     folders = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','space','nothing']
#     for alpha_idx, alpha in enumerate(folders):
#         for row_idx, file in enumerate(os.listdir(f'../data/Train/{alpha}')):
#             if file != '.DS_Store':
#                 image = imread(f'../data/Train/{alpha}/{file}', plugin='pil')
#                 image = image.flatten()
#                 stacked_arrays[row_idx,:image.shape[0],alpha_idx] = image
#     return stacked_arrays

def search_and_destroy():
    folders = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','space','nothing']
    for alpha in folders:
        for file in os.listdir(f'../data/Train/{alpha}'):
            if file == '.ipynb_checkpoints':
                os.rmdir(file)
                print(f'File removed in {alpha}')         
        print(f'Nothing found in {alpha}')

def data_generator():
    '''
    Returns augmented, labeled data
    '''
    train_datagen = ImageDataGenerator(
        rescale= 1./255,
        zoom_range= 0.1,
        height_shift_range= 0.1,
        width_shift_range=0.1,
        rotation_range=20,
        validation_split = 0.2,
        horizontal_flip = True,
        shear_range = 0.1 
        )
    valid_datagen = ImageDataGenerator(rescale= 1./255)
    # test_datagen = ImageDataGenerator(rescale= 1./255)

    train_gen = train_datagen.flow_from_directory(model_params['train_dir'], target_size= (model_params['img_width'], model_params['img_height']), batch_size= model_params['batch_size'])
    valid_gen = valid_datagen.flow_from_directory(model_params['valid_dir'], target_size= (model_params['img_width'], model_params['img_height']), batch_size=model_params['batch_size'])
    # test_gen = test_datagen.flow_from_directory(model_params['test_dir'])
    
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
    model.add(Conv2D(64, (3,3), input_shape=input_shape))
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size= (2,2))) 
    model.add(Dropout(0.2)) 
    # for i, num_filters in enumerate(model_params['filters']):
    #     if i == 0:
    #         model.add(Conv2D(num_filters, (3,3), input_shape= input_shape))
    #     else:
    #         model.add(Conv2D(num_filters, (3,3), activation='relu'))
    #     model.add(MaxPooling2D(pool_size=(2,2)))
    #     model.add(Dropout(0.2))
    model.add(Conv2D(128, (3,3)))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())    
    model.add(MaxPooling2D(pool_size= (2,2)))
    model.add(Dropout(0.2)) 
    model.add(Flatten())
    # model.add(BatchNormalization())
    model.add(Dense(250)) # consider this number - lower potentially
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) 
    # model.add(BatchNormalization())
    model.add(Dense(28))
    model.add(Activation('softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer = Adam(learning_rate = 0.001),
        metrics=['accuracy']
    )

    return model

def model_evaluation_plot():
    '''
    Returns plot of model's performance as measured by accuracy metrics
    '''
    plt.plot(history.history['accuracy'], label= 'accuracy', color='yellowgreen')
    plt.plot(history.history['val_accuracy'], label= 'val_accuracy', color='yellowgreen', linestyle= '--')
    plt.plot(history.history['loss'], label= 'loss', color= 'salmon')
    plt.plot(history.history['val_loss'], label= 'val_loss', color= 'salmon', linestyle= '--')
    plt.xlabel('Epoch')
    plt.ylim([0,1])
    plt.title("Model Performance")
    plt.legend(loc='lower right')
    plt.show()

def confusion_matrix_plot():
    '''
    Returns confusion matrix of model's performance
    '''
    
    pass

if __name__=='__main__':
    model_params = {
        'train_dir': '../data/Train',
        'valid_dir': '../data/Valid',
        'test_dir': '../data/Test',
        'filters': [16, 32, 64],
        'batch_size': 50,
        'img_height': 28, 
        'img_width': 28,
        'epochs': 30 
    }
    # print(img_to_array())
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
    early_stop = EarlyStopping(
        monitor= 'val_loss'

    )
    tensor_checkpoint = ModelCheckpoint('../data/models', save_best_only= True)
    model = cnn_model()
    train_datagen, valid_datagen, train_gen, valid_gen = data_generator()
    model.summary()

    history = model.fit(
        train_gen,
        steps_per_epoch = 58828 // model_params['batch_size'], # updates weight/backprop, can increase (more updates per epoch)
        epochs = model_params['epochs'],
        validation_data = valid_gen,
        # validation_steps = 1, 
        callbacks= [tensorboard_callback, tensor_checkpoint]
    )

    # model.evaluate(valid_gen)
    model_evaluation_plot()
    model.save_weights('v1.h5')