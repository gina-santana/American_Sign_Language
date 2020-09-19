
from skimage.color import rgb2gray
import numpy as np
import cv2
from skimage.io import imread
import matplotlib.pyplot as plt 
from skimage import io, color, filters


if __name__=='__main__':
    image = io.imread('../data/Train/B/B2084.jpg')
    image = rgb2gray(image)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    plt.subplot(2,2,1),plt.imshow(image,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()

