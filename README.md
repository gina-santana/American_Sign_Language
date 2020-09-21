# A Helping Hand: American Sign Language Interpretor

<img src="images/banner.png" width="700" height="317" >

# Table of Contents

- [Background](#background)
- [References](#references)
- [Data](#data)
- [Methods and Models](#methods-and-models)
  + [Model 1](#model-1)
  + [Model 2](#model-2)
  + [Model 3](#model-3)
- [Live Image Classification](#live-image-classification)
- [Conclusion and Next Steps](#conclusion-and-next-steps)

## Background:

Modern American Sign Language (ASL) is a visual language that was developed in 1817 and is used as a visual means of communicating ideas and concepts. It is estimated by the National Center for Health Statistics that there are 28 million Americans living with some degree of hearing loss. Of the 28 million, there is an estimated 2 million Americans who are classified as deaf meaning they are unable to hear everyday sounds/speech even with hearing aids. 10% of the 2 million Americans were born deaf with the remaining 90% developing it later in life. (1) 

<img src="images/alphabet.gif" > 

## Data:

The data I used to train my model consisted of 84,028 total images. I used 28 of the 29 classes from the [original dataset](https://www.kaggle.com/grassknoted/asl-alphabet) meaning I used classes A-Z in addition to the hand gesture indicating a space and images with no hand in the image. I reserved a total of 2,100 images per class for training, 900 images per class for validation and 1 image per class for testing. The images in the dataset featured varied lighting conditions as well as varied distances from the camera (see Figure 1). Because of the variation in lighting and distance, there was very little processing to do on the images. The only image processing used was sobel filtering for edge detection (see Figure 2). 

###### Figure 1:
![raw_data](images/raw_data.png)

###### Figure 2:

<img src="images/B_sobel2.png" width="478" height="407" >

## Methods and Models:

I iterated through 3 versions of my baseline model:

* Model version 1 had a "simple" CNN architecture model
* Model version 2 used transfer learning
* Model version 3 had a "complex" CNN architecture model

Each model was trained twice: once on sobel y-axis filtering and once on sobel x-axis filtering.

### Model 1

This model had 2 main layers that started with convolution. I included additional layers in the mix such as Max pooling, drop out layers in which I gradually increased it from 0.25 to 0.5, batch normalization, as well as 2 dense layers. The last dense layer had 28 neurons for my 28 classes. This particular model had 277,180 total parameters of which 270,716 were trainable. A non-normalized confusion matrix for each model was plotted.

#### Sobel y:

* Accuracy: 93.8%
* loss: 21.89%

###### Figure 3:

<img src="images/model1final.png" width="571" height="457">

###### Figure 4:

<img src="images/cm_modelV1_nonnorm.png" width="816" height="699">

#### Sobel x:

* Accuracy: 97.44%
* loss: 12.66%

###### Figure 5:

<img src="images/model1rerun2.png" width="571" height="457" >

###### Figure 6:

<img src="images/cm_modelv1rerun2.png" width="816" height="699">

### Model 2

Version 2 of the model used transfer learning. The pre-trained CNN model used for this was Xception. Xception is a model that was trained on 350 million images and 17,000 classes and was 71 layers deep. Because of this, I decided to use this particular model. The last 2 layers of the model had global average pooling and a dense layer with my 28 neurons (1 for each class). This version of the model had a total of 20,918,852 parameters, of which 20,864,324 were trainable. (2)(3)

#### Sobel y:

* Accuracy: 90.64%
* Loss: 31.74% 

###### Figure 7:

<img src="images/TransferFinal.png" width="571" height="457">

###### Figure 8:

<img src="images/cm_transfermodel_nonnorm.png" width="816" height="699">

#### Sobel x:

* Accuracy: 90.73%
* Loss: 43.45%

###### Figure 9:

<img src="images/transferrerun.png" width="571" height="457">

###### Figure 10:

<img src="images/transferrerun_cm.png" width="816" height="699">

### Model 3:

Due to the undesirable results from model version 2, I built a new model from scratch without transfer learning. This 3rd and final version had more layer and more trainable parameters compared to version 1. This model only had layers for convolution, drop out which I kept at a consistent 0.5, and 2 dense layers. The total number of parameters equated to 3,230,108 of which all were trainable.

#### Sobel y:

* Accuracy: 95.96%
* Loss: 12.10%

###### Figure 11:

<img src= "images/ModelV3Final.png" width="571" height="457">

###### Figure 12:

<img src="images/cm_modelv3_nonnorm.png" width="816" height="699">

#### Sobel x:

* Accuracy: 97.21%
* Loss: 8.35%

###### Figure 13:

<img src="images/modelv3rerun.png" width="571" height="457">

###### Figure 14:

<img src="images/cm_modelv3rerun.png" width="816" height="699">

The following chart is a summary of each model's performance including accuracy, loss, number of parameters, and the most confused for classes:

![summary](images/summary.png)

### Live Image Classification

I took my best performing model (in this case, version 3 using sobel x-axis filter) and tested it's performance when it came to live image classification. I set up a camera to show me in real-time what letter was represented by my hand gesture. Unfortunately, it took a bit of repositioning of the camera to find an ideal background for my hand. This demonstrates that further image processing may be needed in order to increase the generalizability of backgrounds. The following is a snippet of that demonstration:

![live_demo](images/ezgif-2-48df50ed627e.gif)

It was able to accurately translate what I was trying to spell out in sign language (G-I-N-A). The application of this type of live image classification for sign language could help to promote communication and have real-time translations for those unfamiliar with sign language. 

### Conclusion and Next Steps

It was exciting to learn how to sign my own name and have the model be able to translate it in real-time. Sobel x-axis filtering performed better in general across all 3 version of my model so this may be the optimal filter axis to use for hand gesture-related image data. For my next steps I want to experiment with varied backgrounds and/or other image processing techniques to help the model distinguish between gestures that look very similar. As we saw with the confusion matrices, the classes that were confused most often looked very similar to each other. For example, 'V' and 'U'.

## References:

* (1) American Sign Language [website] https://www.startasl.com/american-sign-language/
* (2) Chollet F. Xception: Deep Learning with Depthwise Separable Convolutions. https://arxiv.org/abs/1610.02357
* (3) Xception [website] https://www.mathworks.com/help/deeplearning/ref/xception.html

