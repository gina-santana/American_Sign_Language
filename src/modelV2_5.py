from cnn_model import CNNModel
from helper_functions import search_and_destroy
from keras.utils import plot_model


model_params = {
    'train_dir': '../../../test2/Training',
    'valid_dir': '../../../test2/Validation',
    'test_dir': '../data/Test',
    'model_type': 'transfer',
    'sobel_axis': 'x', # 'x' or 'y'
    'batch_size': 64,
    'img_height': 150, 
    'img_width': 150,
    'epochs': 5
}

search_and_destroy()
model = CNNModel(**model_params)
# model.load_weights_from_file('../model_weights/v2transferrerun.h5')
# layers = model.trained_model.layers
model.fit()
# model.save_model_to_file()

evaluated_validation = model.evaluate_model(model.create_generator(model.valid_dir, shuffle=False))
model.print_classification_report(evaluated_validation)
model.plot_confusion_matrix(**evaluated_validation)

# plot_model(model.trained_model, to_file='../images/V2structure.png', show_shapes=True, show_layer_names=True)

# model.save_weights('v2transfer.h5')
# model.save_weights('v2transferrerun.h5')