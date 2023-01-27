# Wheat Head Auto Counter

<br>

Wheat head density is the number of wheat heads per unit ground area. It's an important measure that researchers use to determine wheat yield. To get this value researchers have to count the number of wheat heads on wheat field images. Manually counting wheat heads is tedious. This desktop app uses computer vision to automatically detect and count wheat heads on images of wheat fields.

The app ouputs an interactive image. Clicking on a dot converts it into a bounding box. This feature plus image zoom makes it easier for a person to review the output image. This a flask app running on the desktop. Internally the app is powered by a Yolov5m model that was trained on data from the Global Wheat Head Dataset 2021.
<br>

<br>
<img src="https://github.com/vbookshelf/Wheat-Head-Auto-Counter/blob/main/images/wheat-app-image.png" height="400"></img>
<i>Sample prediction<br>Clicking on a dot converts it into a bounding box</i><br>
<br>

The model has a map@0.5 of 0.93.
The validation count error varied by domain. A domain is a combination of the place where the wheat photos were taken and the wheat development stage. There are 47 domains in the dataset.  32 domains had count errors less than 10 percent. 41 domains had count errors less than 20 percent.

Having a human in the loop would be the best way to use this app. For each prediction, a person should look at the dots and their associated bounding boxes, then adjust the count up or down to arrive at the final number of wheat heads.

<br>

## Demo

<br>
<img src="https://github.com/vbookshelf/Wheat-Head-Auto-Counter/blob/main/images/wheat-app-gif.gif" height="450"></img>
<i>Demo showing what happens after a user submits three wheat images</i><br>
<br>


<br>

## 1- Main Features

- Draws dots on detected wheat heads.
- Clicking on a dot converts it into a bounding box.
- A user can zoom into an image by using the desktop zoom feature that’s built into Mac and Windows.
- Multiple images can be submitted
- Free to use. Free to deploy. No server rental costs like with a web app.
- Runs locally without needing an internet connection

<br>

## 2- Cons

- It’s not a one click setup. The user needs to have a basic knowledge of how to use the command line to set up a virtual environment, download requirements and launch a python app.
- The model’s ability to generalize is unproven. The dataset includes images from 22 locations around the world. I’m uncertain of how this model will perform on images from locations that are not represented in the Global Wheat Head Dataset.

<br>

## 3- How to zoom into the image

To magnify the image use the desktop zoom feature that’s built into both Mac and Windows 10. 

Place the mouse pointer on the area that you want to magnify then:
- On Mac, move two fingers apart on the touchpad
- On Windows, hold down the windows key and press the + key

<br>

## 4- How to run this app

### First download the project folder from Kaggle

I've stored the project folder (named wheat-head-auto-counter) in a Kaggle dataset.<br>
https://www.kaggle.com/datasets/vbookshelf/wheat-head-auto-counter


I suggest that you download the project folder from Kaggle instead of from this GitHub repo. This is because the project folder on Kaggle includes the trained model. The project folder in this repo does not include the trained model because GitHub does not allow files larger than 25MB to be uploaded.<br>
The model is located inside a folder called TRAINED_MODEL_FOLDER, which is located inside the yolov5 folder:<br>
wheat-head-auto-counter/yolov5/TRAINED_MODEL_FOLDER/

<br>

### System Requirenents

You'll need about 1.3GB of free disk space. Other than that there are no special system requirements. The app will run on a CPU. I have an old 2014 Macbook Pro laptop with 8GB of RAM. This app runs on it without any issues.


<br>

### Overview

This is a standard flask app. The steps to set up and run the app are the same for both Mac and Windows.

1. Download the project folder.
2. Use the command line to pip install the requirements listed in the requirements.txt file. (It’s located inside the project folder.) 
3. Run the app.py file from the command line.
4. Copy the url that gets printed in the console.
5. Paste that url into your chrome browser and press Enter. The app will open in the browser.

This app is based on Flask and Pytorch, both of which are pure python. If you encounter any errors during installation you should be able to solve them quite easily. You won’t have to deal with the package dependency issues that happen when using Tensorflow.

<br>

### Detailed setup instructions

The instructions below are for a Mac. I didn't include instructions for Windows because I don't have a Windows pc and therefore, I could not test the installtion process on windows. If you’re using a Windows pc then please change the commands below to suit Windows. 

You’ll need an internet connection during the first setup. After that you’ll be able to use the app without an internet connection.

If you are a beginner you may find these resources helpful:

The Complete Guide to Python Virtual Environments!<br>
Teclado<br>
(Includes instructions for Windows)<br>
https://www.youtube.com/watch?v=KxvKCSwlUv8&t=947s

How To Create Python Virtual Environments On A Mac<br>
https://www.youtube.com/watch?v=MzuGMSw8la0&t=167s

<br>

```

1. Download the project folder, unzip it and place it on your desktop.
In this repo the project folder is named: wheat-head-auto-counter
Then open your command line console.
The instructions that follow should be typed on the command line. 
There’s no need to type the $ symbol.

2. $ cd Desktop

3. $ cd project_folder

4. Create a virtual environment. (Here it’s named myvenv)
This only needs to be done once when the app is first installed.
You'll need to have python3.8 available on your computer.
When you want to run the app again you can skip this step.
$ python3.8 -m venv myvenv

5. Activate the virtual environment
$ source myvenv/bin/activate

4. Install the requirements.
This only needs to be done once when the app is first installed.
When you want to run the app again you can skip this step.
$ pip install -r requirements.txt

5. Launch the app.
This make take a few seconds the first time.
$ python app.py

6. Copy the url that gets printed out (e.g. http://127.0.0.1:5000)

7. Paste the url into your chrome browser and press Enter. The app will launch in the browser. 

8. To stop the app type ctrl C in the console.
Then deactivate the virtual environment.
$ deactivate

```

There are sample images in the sample_wheat_images folder. You can use them to test the app.

While the app is analyzing, please look in the console to see if there are any errors. If there are errors, please do what’s needed to address them. Then relaunch the app.

<br>


## 5- Model Training and Validation

The model card contains a summary of the training and validation datasets as well as the validation results. There's also some info about the app. Please refer to this document:<br>
https://github.com/vbookshelf/Wheat-Head-Auto-Counter/blob/main/wheat-head-auto-counter-v1.0/Model-Card-and-App-Info%20v1.0.pdf

All the project jupyter notebooks are stored in the folder called "Notebooks". There are four notebooks. 
Each notebook was run either on Kaggle or on VAST.<br>
https://github.com/vbookshelf/Wheat-Head-Auto-Counter/tree/main/wheat-head-auto-counter-v1.0/Notebooks


Exp_05-Kaggle<br>
The code to create 7 folds. Only fold 0 was used for training and validation.

Exp_07-VAST<br>
The code for training and validating the model.

Exp_09-Kaggle<br>
The code for reviewing the val preds made by the model created in exp07. Demonstrates GPU nference using the Yolov5 detect.py workflow.

Exp_11-Kaggle<br>
The code for reviewing the val preds made by the model created in exp07. Demonstrates cpu inference using the Torch Hub workflow. This is the inference method that's used in the app.


<br>

## 6- Licenses

All code that I have created is free to use under an MIT license.
 
The dataset used to train the model is available under a Creative Commons Attribution 4.0 International Public License.<br>
https://creativecommons.org/licenses/by/4.0/legalcode

The Ultralytics Yolov5 model is licensed under a GPL-3.0 License.<br>
https://github.com/ultralytics/yolov5/blob/master/LICENSE

<br>

## 7- Citations

DAVID Etienne. (2021). Global Wheat Head Dataset 2021 (1.0) [Data set]. Zenodo.<br>
https://doi.org/10.5281/zenodo.5092309

<br>

## 8- Acknowledgements

Many thanks to Kaggle for the free GPU and other great resources they continue to provide.

I also want to thank the GWHD team for the dataset that they’ve generously made publicly available.

Many thanks to the team at Ultralytics for the Yolov5 model and pre-trained weights they’ve made freely available.

<br>

## 9- References and Resources

Paper:<br>
Global Wheat Head Dataset 2021: more diversity to improve the benchmarking of wheat head localization methods<br>
https://arxiv.org/abs/2105.07660

Dataset on Zenodo<br>
https://zenodo.org/record/5092309#.Y7jTtuxBzUI

Dataset on Kaggle<br>
https://www.kaggle.com/datasets/vbookshelf/global-wheat-head-dataset-2021

Ultralytics Yolov5<br>
https://github.com/ultralytics/yolov5

The Complete Python Course | Learn Python by Doing in 2022<br>
Udemy<br>
https://www.udemy.com/course/the-complete-python-course/

Flask experiments<br>
https://github.com/vbookshelf/Flask-Experiments

W3.CSS Tutorial<br>
https://www.w3schools.com/w3css/defaulT.asp

