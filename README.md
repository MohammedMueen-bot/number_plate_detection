# NUMBER PLATE RECOGNITION PROJECT
   In this project i have used Inception-ResNet-v2 a convolutional neural network to train a custom object detector model on number plate dataset  and integrated the model with Flask.the model predicts the number plate then region of interest is cropped using opencv and then using tesseract we extract the text.
   # How to run this project.
    1) download and extract the project zip file or clone the repo
    2) download the model file provided below and move to Number_plate_detection\static\models folder
    3) using command prompt run app.py file   
   ([download model](https://drive.google.com/file/d/1lPlrW5YwCgmRISVdcWjFkPUSpMTsUUTZ/view?usp=sharing))


   install all the necessary packages in your virtual environment 
   - Python 3.9.5
   - Flask 2.0.2
   - tesseract 3.02
   - tensorflow 2.6.0
   - OpenCv 4.5.3
