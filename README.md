# Wheat Head Auto Counter
This is a free desktop wheat head counting tool that uses computer vision to detect and count wheat heads on images of wheat fields. It’s a flask app running on the desktop. Internally the app is powered by a Yolov5m model that was trained on data from the Global Wheat Head Dataset 2021.

<br>
<img src="https://github.com/vbookshelf/Wheat-Head-Auto-Counter/blob/main/images/wheat-app-image.png" height="400"></img>
<i>Sample prediction<br>Clicking on a dot converts it into a bounding box</i><br>
<br>

The model has a map@0.5 of 0.93.
The validation count error varied by domain. A domain is a combination of the place where the wheat photos were taken and the wheat development stage. There are 47 domains in the dataset.  32 domains had count errors less than 10 percent. 42 domains had count errors less than 20 percent.

Having a human in the loop would be the best way to use this app. For each prediction, a person should look at the dots and their associated bounding boxes, then adjust the count up or down to arrive at the final number of wheat heads. The workflow is not entirely “hands-free” but it’s still much faster and less tedious than manual counting from scratch.

<br>

## Demo

<br>
<img src="https://github.com/vbookshelf/Wheat-Head-Auto-Counter/blob/main/images/wheat-app-gif.gif" height="450"></img>
<i>Demo showing what happens after a user submits three wheat images</i><br>
<br>



<br>
