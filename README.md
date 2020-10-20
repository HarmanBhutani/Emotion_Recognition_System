# Emotion Recognition System

*Author* : - **Harman Bhutani**

Aim of the Project
------------------

Human facial expressions can be easily classified into 7 basic emotions: happy, sad, surprise, fear, anger, disgust, and neutral. Our facial emotions are expressed through activation of specific sets of facial muscles. These sometimes subtle, yet complex, signals in an expression often contain an abundant amount of information about our state of mind. Through facial emotion recognition, we are able to measure the effects that content and services have on the audience/users through an easy and low-cost procedure.

Database
--------

The dataset we used for training the model is from a Kaggle Facial Expression Recognition Challenge (FER2013)

It comprises a total of 35887 pre-cropped, 48-by-48-pixel grayscale images of faces each labeled with one of the 7 emotion classes: anger, disgust, fear, happiness, sadness, surprise, and neutral.

Dependencies
------------

* NumPy
* Tensorflow
* TFLearn
* OpenCV

Usage
-----

To train the model

`python emotion_recognition_system.py train`

to run the main program

`python emotion_recognition_system.py run`
