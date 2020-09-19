import pygame
import pygame.camera
from pygame.font import font
from pygame.locals import *
from time import sleep
from tensorflow.keras.models import load_model
from skimage import filters, color, io
from skimage.transform import rotate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import threadingfrom queue import LifoQueue

def transform(image):
    pass

if __name__=='__main__':
    class_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','space','nothing']
    model = load_model('v2.h5')
    display_width = 200
    display_height = 200
    pygame.init()
    pygame.camera.init()
    camlist = pygame.camera.list_cameras()
    if camlist:
        cam = pygame.camera.Camera(camlist[0], (640,480))
    cam.start()
    gameDisplay = pygame.display.set_model((display_width, display_height))
    black = (0, 0, 0)
    white = (255, 255, 255)
    font = Font('freesanbold.ttf', 18)

    clock = pygame.time.Clock()
    while True:
        image_surface = cam.get_image()
        image_arr = transform(image_surface)
        probabilities = model.predict(image_arr[1])
        text = class_labels[np.argmax(probabilities)]
        text_surface = font.render(text, True, black)
        textrect = text_surface.get_rect()
        textrect.center = 40, 20
        gameDisplay.fill(white)
        gameDisplay.blit(image_surface, (0,0))
        gameDisplay.blit(text_surface, textrect)
        pygame.display.update()
        clock.tick(60)
        model.reset_states()