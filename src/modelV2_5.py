from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

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

    train_gen = train_datagen.flow_from_directory('../data/Train', target_size= (150, 150), batch_size= 50)
    valid_gen = valid_datagen.flow_from_directory('../data/Valid', target_size= (150, 150), batch_size= 50)
    
    return train_datagen, valid_datagen, train_gen, valid_gen

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

def create_transfer_model(input_size, n_categories, weights = 'imagenet'):
        # note that the "top" is not included in the weights below
        base_model = Xception(weights=weights,
                          include_top=False,
                          input_shape=input_size)
        
        model = base_model.output
        model = GlobalAveragePooling2D()(model)
        predictions = Dense(n_categories, activation='softmax')(model)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model

def print_model_properties(model, indices = 0):
     for i, layer in enumerate(model.layers[indices:]):
        print(f"Layer {i+indices} | Name: {layer.name} | Trainable: {layer.trainable}")

def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True

if __name__=='__main__':
    model = create_transfer_model(input_size=(150,150,3), n_categories=28, weights='imagenet')
    # model.summary()
    change_trainable_layers(model,131)
    # print_model_properties(model)
    train_datagen, valid_datagen, train_gen, valid_gen = data_generator()
    model.compile(
        loss='categorical_crossentropy',
        optimizer = Adam(learning_rate = 0.1),
        metrics=['accuracy']
    )
    
    tensor_checkpoint = ModelCheckpoint('../data/models', save_best_only= True)
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

    history = model.fit(
        train_gen,
        steps_per_epoch = 58828 // 50,
        epochs = 30,
        validation_data = valid_gen,
        validation_steps = 1,
        callbacks= [tensorboard_callback, tensor_checkpoint]
    )
    
    model_evaluation_plot()

    model.save_weights('transfer.h5')
    
    


