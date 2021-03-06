from helper_functions import search_and_destroy
from cnn_model import CNNModel
from keras.utils import plot_model

model_params = {
    'train_dir': '../../../test2/Training',
    'valid_dir': '../../../test2/Validation',
    'test_dir': '../data/Test',
    'model_type': 'complex',
    'sobel_axis': 'x', # 'x' or 'y'
    'batch_size': 64,
    'img_height': 64, 
    'img_width': 64,
    'epochs': 5
}

search_and_destroy()
model = CNNModel(**model_params)
# model.load_weights_from_file('../model_weights/v3rerun.h5')
# layers = model.trained_model.layers
model.fit()
model.save_model_to_file()

evaluated_validation = model.evaluate_model(model.create_generator(model.valid_dir, shuffle=False))
model.print_classification_report(evaluated_validation)
model.plot_confusion_matrix(**evaluated_validation)

# plot_model(model.trained_model, to_file='../images/V3structure.png', show_shapes=True, show_layer_names=True)

# model.save_weights('v3.h5') # sobel_y
# model.save_weights('v3rerun.h5') # sobel_x

