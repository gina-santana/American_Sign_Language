import cv2
import numpy as np
import pygame
import pygame.camera
from pygame.font import Font
from tensorflow.keras.models import load_model
from skimage.transform import rotate
from cnn_model import CNNModel
 
model_params = {
   'train_dir': '../data/Train',
   'valid_dir': '../data/Valid',
   'test_dir': '../data/Test',
   'model_type': 'complex',
   'batch_size': 64,
   'img_height': 64,
   'img_width': 64,
   'epochs': 5,
   'sobel_axis': 'x'
}

class_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S',
   'T','U','V','W','X','Y','Z','space','nothing']

blue = (0, 0, 255)
white = (255, 255, 255)
display_width = 400
display_height = 400
 

def transform_image(image):
   scaled_image = pygame.transform.smoothscale(image, (64, 64))
   rotated_image = pygame.transform.rotate(scaled_image, 90)
   img_array = pygame.surfarray.array3d(rotated_image)
 
   sobel = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize = 5)
   return np.expand_dims(np.reshape(sobel, (64, 64, 3)), axis = 0)
 
 
if __name__ == "__main__":
   model = CNNModel(**model_params)
   model.load_weights_from_file('../model_weights/v3rerun.h5')
 
   pygame.init()
   pygame.camera.init()
   camlist = pygame.camera.list_cameras()
   if camlist:
       cam = pygame.camera.Camera(camlist[0],(640,480)) 
   cam.start()
   gameDisplay = pygame.display.set_mode((display_width, display_height))
   font = Font('freesansbold.ttf', 45)
 
   clock = pygame.time.Clock()
   while True:
       image_sur = cam.get_image()
       image = transform_image(image_sur)
       probabilities = model.trained_model.predict(image)
       text = class_labels[np.argmax(probabilities)]
       textsurf = font.render(text, True, blue)
       textrect = textsurf.get_rect()
       textrect.center = 200, 50
       gameDisplay.fill(white)
       gameDisplay.blit(image_sur, (0,0))
       gameDisplay.blit(textsurf, textrect)
       pygame.display.update()
       clock.tick(60)
       model.trained_model.reset_states()
