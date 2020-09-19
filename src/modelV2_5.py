from cnn_model import CNNModel
from helper_functions import search_and_destroy


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
model.fit()
# model.save_model_to_file()

evaluated_validation = model.evaluate_model(model.create_generator(model.valid_dir, shuffle=False))
model.print_classification_report(evaluated_validation)
model.plot_confusion_matrix(**evaluated_validation)

# model.save_weights('v2transfer.h5')
# model.save_weights('v2transferrerun.h5')