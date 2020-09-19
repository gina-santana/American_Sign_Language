from cnn_model import CNNModel
from helper_functions import search_and_destroy

model_params = {
    'train_dir': '../../../test2/Training',
    'valid_dir': '../../../test2/Validation',
    'test_dir': '../data/Test',
    'sobel_axis': 'x', # 'x' or 'y'
    'model_type': 'simple',
    'batch_size': 64,
    'img_height': 64, 
    'img_width': 64,
    'epochs': 5
}

search_and_destroy()
model = CNNModel(**model_params)
model.fit()

evaluated_validation = model.evaluate_model(model.create_generator(model.valid_dir, shuffle=False))
model.print_classification_report(evaluated_validation)
model.plot_confusion_matrix(**evaluated_validation)

# model.save_weights('v1rerun.h5') #sobel_y
# model.save_weights('v1rerun2.h5') # sobel_x

