
"""
NOTES

1- Yolo version being used: Yolov5-5.0
The yolov5-5.0 folder has been named yolov5.

2- Model version being used: exp7_best.pt

3- The trained model has been placed here:
yolov5/TRAINED_MODEL_FOLDER/

4- In this app the endpoints send and receive data using Ajax.
This allows the web pages to be updated without being refreshed each time.
This is a simple example that shows how the process works:
https://github.com/vbookshelf/Flask-Experiments/tree/main/Exp_11-working-ajax-flask-request-response-example-template

5- I changed line 173 in detect.py (located inside the yolov5 folder) to include 'PyYAML' in the list
of packages that Yolo should not check for.
This fixed an error in PyCharm.
Line 73: check_requirements(exclude=('pycocotools', 'thop', 'PyYAML'))

6- This version uses Torch Hub for inference and not detect.py

7- By design this app just freezes when there's an error.
I've not included a try block to catch errors. This is so that all errors
will be clearly shown in the console.
This will help users to find and fix errors faster.

8- Because this app is using Torch hub for inference, there's no need to specify the device.
The app should use the GPU if available.
If there's no GPU it will use the CPU.

"""


# Note: In flask it's common practice to name this file utils.
# But doing that will create an error because yolov5 also has a utils package.
# See: https://github.com/ultralytics/yolov5/issues/1181
# There here this file is named my_utils.
from my_utils import *

import os
from flask import Flask, render_template, url_for, request, redirect, jsonify
from werkzeug.utils import secure_filename

# Specify the model name/s here.
MODEL_LIST = ['exp7_best.pt']



# Create an instance of the Flask class
app = Flask(__name__, static_url_path='/static')

# Note that there is code below that converts all file extensions to lower case.
app.config['ALLOWED_EXTENSIONS'] = ['.png', '.jpg', '.jpeg']

# Get the absolute path to the folder called 'static'.
# We must get this path before we change the working directory.
ABS_PATH_TO_STATIC = os.path.abspath("static")


# This endpoint loads the index.html page.
@app.route('/')
def home_func():

    # On page load delete any previous user submitted data
    # -----------------------------------------------------

    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('uploads') == True:
        shutil.rmtree('uploads')

    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('yolov5/png_images_dir') == True:
        shutil.rmtree('yolov5/png_images_dir')


    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('yolov5/pred_images_dir') == True:
        shutil.rmtree('yolov5/pred_images_dir')


    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('static/pred_images_dir') == True:
        shutil.rmtree('static/pred_images_dir')

    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('static/analysis_images_dir') == True:
        shutil.rmtree('static/analysis_images_dir')

    print('Previous image data deleted.')

    return render_template('index.html')


# This is the endpoint that loads the page that displays the model card.
@app.route('/about')
def about_func():
    return render_template('more-info.html')


# This is the endpoint that loads the page that displays the FAQ.
@app.route('/faq')
def faq_func():
    return render_template('faq.html')



# This endpoint contains the code for:
# - image file uploading
# - image processing
# - inference
# - displaying the inference results on the page
# The code refers to the images as png images, but they can be
# either png or jpg.
@app.route('/upload_ajax', methods=['POST'])
def upload_ajax():

    # This try except block is commented out.
    #try:

    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('uploads') == True:
        shutil.rmtree('uploads')

    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('yolov5/png_images_dir') == True:
        shutil.rmtree('yolov5/png_images_dir')

    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('static/pred_images_dir') == True:
        shutil.rmtree('static/pred_images_dir')

    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('yolov5/pred_images_dir') == True:
        shutil.rmtree('yolov5/pred_images_dir')


    # Create a new folder to store uploaded dicom files
    if os.path.isdir('uploads') == False:
        uploads = 'uploads'
        os.mkdir(uploads)



    # Get a list.
    # my_files is the name that is used in the html code.
    file_list = request.files.getlist('my_files')

    print(file_list)

    for item in file_list:
        # Get the file name.
        # We need to secure and clean up the file name.
        fname = item.filename

        # Get the file extension e.g. .dicom
        # Convert to lower case
        extension = os.path.splitext(item.filename)[1].lower()

        # Only save the file if it has a .dicom extension
        if extension in app.config['ALLOWED_EXTENSIONS']:

            # Create a secure file name.
            # Replace any spaces with underscores.
            # Any other malicious symbols get removed.
            fname = secure_filename(fname)

            # Save the file to a folder called uploads only if it has a .dicom extension
            # We need to create a folder called uploads.
            item.save(f'uploads/{fname}')


    # Get a list of files in the uploads folder
    upfile_list = os.listdir('uploads')


    # Get only list items that are dicom files.
    # Sometimes there are strange hidden files in local folders

    image_list = []

    for item in upfile_list:

        # Get the file extension and convert to lower case.
        file_ext = item.split('.')[1].lower()
        file_ext_with_dot = f'.{file_ext}'

        if file_ext_with_dot in app.config['ALLOWED_EXTENSIONS']:
            image_list.append(item)


    # MAIN FUNCTIONS
    # --------------


    # Transfer the uploaded images to png_images_dir.
    # image_list is a list of the file names of the images that were uploaded.
    # image_size_list is a list of all the uploaded image sizes [(height, width), (height, width),...]
    # image_size_list has the same image order as image_list
    image_size_list = transfer_images_to_folder(image_list)


    # Make a prediction on all images in png_images_dir.
    # Create and save a csv file called df_fin_preds.csv
    # The csv file contains all the predictions and info on the images e.g. fname
    # image_list is a list of the file names of the images that were uploaded.
    # image_size_list is a list of all the image sizes [(height, width), (height, width),...]
    df_fin = predict_on_all_uploaded_images(MODEL_LIST, image_list, image_size_list)


    # Draw dots on the images.
    # Returns a dict containing the number of dots on each image.
    # image_list is a list of the file names of the images that were uploaded.
    num_preds_dict = process_torch_hub_predictions(image_list, ABS_PATH_TO_STATIC)


    # ---------------


    # Create html for the images that will be loaded into the hidden image elements.
    # This will cache the images and make them instantly available when a user clicks on a link
    # to display an image.
    for i, item in enumerate(image_list):


        if i == 0:
            image_fin_str = f"""<img class="w3-round unblock" src="/static/pred_images_dir/{item}"  height="580">"""
        else:
            image_fin_str = image_fin_str + f"""<img  class="w3-round unblock" src="/static/pred_images_dir/{item}"  height="580">"""


    # Create the html for the clickable links to images.
    start_str = "<ul>"
    for i, item in enumerate(image_list):

        # Create the string that shows the number of bboxes
        num_preds = num_preds_dict[item]

        if num_preds == 1:
            num_str = str(num_preds) + ' ' + 'wheat head'
        else:
            num_str = str(num_preds) + ' ' + 'wheat heads'


        if i == 0:
            fin_str = start_str + f'<li class="row w3-text-black w3-border-right w3-border-black w3-padding-bottom" onclick="ajaxGetFilename(this.innerHTML)"><a href="#"><b>{num_str}</b><br>{item}</a></li>'
        else:
            fin_str = fin_str + f'<li class="row w3-padding-bottom" onclick="ajaxGetFilename(this.innerHTML)"><a href="#"><b>{num_str}</b><br>{item}</a></li>'

    html_str = fin_str + '</ul>' + """<script>jQuery('li').click(function(event){
                //remove all pre-existing active classes
                jQuery('.row').removeClass('w3-text-black w3-border-right w3-border-black');
        
                //add the active class to the link we clicked
                jQuery(this).addClass('w3-text-black w3-border-right w3-border-black');
                event.preventDefault();
                 });</script>"""


    first_fname = image_list[0]

    main_image_str = f"""<img id="selected-image"  onclick="get_click_coords(event, this.src)" class="w3-round unblock" src="/static/pred_images_dir/{first_fname}"  height="580" alt="Wheat">"""

    output_response = {"html_str": html_str, "main_image_str": main_image_str, "image_fin_str": image_fin_str}

    # Delete the data the user has submitted.
    # The png images in static/pred_image_dir don't get deleted here because
    # the app needs to display these images on the main page.
    # The static/pred_image_dir folder gets deleted each time the user submits new files.
    #delete_user_submitted_data()

    return jsonify(output_response)

"""
    except:

        output_response = { "html_str": '<p>Error. Please look at the console for more info.<br>Please submit only dicom files.<br> Allowed extensions: .png, .jpg, .jpeg</p>'}

        # Delete the data the user has submitted.
        # The png images in static/pred_image_dir don't get deleted here because
        # the app needs to display these images on the main page.
        # The static/pred_image_dir folder gets deleted each time the user submits new files.
        delete_user_submitted_data()

        return jsonify(output_response)

"""


# When the user clicks a file name,
# display that image as the main image on the page.
@app.route('/process_ajax', methods=['POST'])
def process_ajax():

    # Get the value of the 'file_name' key
    # Example fname: 0 wheat heads<br>47c8858666bcce92bcbd57974b5ce522.png</a>
    fname = request.form.get('file_name')

    #print(fname)

    # Remove the first part of the str to get this fname format:
    # 47c8858666bcce92bcbd57974b5ce522.png</a>
    fname = fname.split('<br>')[1]


    # **** The next line needs to eb fixed. The user can also upload png images.
    # Therefore, we shouldn't replace the image extension.

    # Replace </a> with nothing
    # 47c8858666bcce92bcbd57974b5ce522.png</a>
    image_fname = fname.replace("</a>", "")

    print(image_fname)


    # Create html code containing the image info
    info_in_html = f"""<img id="selected-image"  onclick="get_click_coords(event, this.src)" class="w3-round unblock" src="/static/pred_images_dir/{image_fname}"  height="580" alt="Wheat">"""

    output = {"output1":  info_in_html}

    return jsonify(output)



# When the user clicks a file name,
# display that image as the main image on the page.
@app.route('/process_sample_ajax', methods=['POST'])
def process_sample_ajax():

    # Get the value of the 'file_name' key
    # Example fname: 0 results<br>47c8858666bcce92bcbd57974b5ce522.dicom
    fname = request.form.get('file_name')


    # Remove the first part of the str to get this fname format:
    # 47c8858666bcce92bcbd57974b5ce522.dicom</a>
    fname = fname.split('<br>')[1]


    # Replace .dicom with .png
    # 47c8858666bcce92bcbd57974b5ce522.png
    image_fname = fname.replace("</a>", "")


    print(image_fname)

    # Create html code containing the image info
    info_in_html = f"""<img id="selected-image" class="w3-round unblock" src="/static/sample_images/{image_fname}"  height="580" alt="Wheat">"""

    output = {"output1":  info_in_html}

    return jsonify(output)


# When the user clicks on a dot on an image
# this endpoint converts that dot into a bounding box.
@app.route('/process_click_info', methods=['POST'])
def process_click_info():

    # Get the image path
    fname = request.form.get('fname')

    # Get the last item in the list
    # Example: [http://127.0.0.1:5000/static/pred_images_dir/f23b18fec1.png]
    image_fname = fname.split('/')[-1:][0]

    # Convert the dot into a bounding box
    output = replace_dot_with_bbox(image_fname, dot_width=10)

    #return ("", 204)
    return jsonify(output)



# This endpoint is used to test that the app is working
@app.route('/test')
def test():
    return 'This is a test...'




if __name__ == '__main__':
    app.run()