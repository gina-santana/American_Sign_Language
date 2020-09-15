from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.metrics import plot_confusion_matrix

def data_generator():
    '''
    Returns augmented, labeled data
    '''
    train_datagen = ImageDataGenerator(
        rescale= 1./255, 
        validation_split = 0.2,
        vertical_flip = True,
        horizontal_shift_range = 0.1,
        shear_range = 0.2 
        )
    valid_datagen = ImageDataGenerator(rescale= 1./255)
    test_datagen = ImageDataGenerator(rescale= 1./255)

    train_gen = train_datagen.flow_from_directory(model_params['train_dir'], target_size= (model_params['img_width'], model_params['img_height']))
    valid_gen = valid_datagen.flow_from_directory(
        model_params['valid_dir'],
        target_size= (model_params['img_width'], model_params['img_height']), 
        shuffle = True
    )
    test_gen = test_datagen.flow_from_directory(model_params['test_dir'])
    
    return (train_datagen, valid_datagen, test_datagen, train_gen, valid_gen, test_gen)

def cnn_model():
    '''
    Returns Convolutional Neural Network model arcitecture 
    '''
    if K.image_data_format() == 'channels_first':
        input_shape = (3, model_params['img_width'], model_params['img_height'])
    else:
        input_shape = (model_params['img_width'], model_params['img_height'], 3)
    
    model = Sequential()
    for i, num_filters in enumerate(model_params['filters']):
        if i == 0:
            model.add(Conv2D(num_filters, (3,3), input_shape= input_shape))
        else:
            model.add(Conv2D(num_filters, (3,3)))
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size= (2,2))) 

    model.add(Flatten())
    model.add(Dense(29)) 
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(26))
    model.add(Activation('softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer = Adam(),
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
        'img_height': 200 ,
        'img_width': 200 ,
        'epochs': 100 
    }

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
    train_datagen, valid_datagen, test_datagen, train_gen, valid_gen, test_gen = data_generator()
    model.summary()

    history = model.fit(
        train_gen,
        steps_per_epoch = 3,
        epochs = model_params['epochs'],
        validation_data = valid_gen,
        validation_steps = 1,
        callbacks= [tensor_checkpoint]
    )

    model.evaluate(valid_gen)
    print()
    model_evaluation_plot()
    model.save_weights('v1.h5')
