
import numpy as np
import pandas as pd
import os
import cv2
import shutil
from flask import request

import torch
import torchvision



# Draw the boxes on the image
def draw_bbox(image, xmin, ymin, xmax, ymax, text=None, line_thickness=20):

    """
    Set text=None to only draw a bbox without
    any text or text background.
    E.g. set text='Balloon' to write a
    title above the bbox.

    Output:
    Returns an image with one bounding box drawn.
    The title is optional.
    To draw a second bounding box pass the output image
    into this function again.

    """

    w = xmax - xmin
    h = ymax - ymin

    # Draw the bounding box
    # ......................

    start_point = (xmin, ymin)
    end_point = (xmax, ymax)
    bbox_color = (255, 255, 255)
    bbox_thickness = line_thickness

    image = cv2.rectangle(image, start_point, end_point, bbox_color, bbox_thickness)

    # Draw the background behind the text
    # ....................................

    # Only do this if text is not None.
    if text:
        # Draw the background behind the text
        #text_bground_color = (0, 0, 0)  # black
        #cv2.rectangle(image, (xmin, ymin - 150), (xmin + w, ymin), text_bground_color, -1)

        # Draw the text
        text_color = (255, 255, 255)  # white
        font = cv2.FONT_HERSHEY_DUPLEX
        origin = (xmin, ymin - 30)
        fontScale = 3
        thickness = 10

        image = cv2.putText(image, text, origin, font,
                            fontScale, text_color, thickness, cv2.LINE_AA)

    return image



# Draw the square dots on the image
def draw_square_dot(image, xmin, ymin, xmax, ymax, dot_width, text=None, line_thickness=20):
    """
    Set text=None to only draw a bbox without
    any text or text background.
    E.g. set text='Balloon' to write a
    title above the bbox.

    Output:
    Returns an image with one dot drawn at the bbox center.
    The title is optional.
    To draw a second bounding box pass the output image
    into this function again.

    """

    w = xmax - xmin
    h = ymax - ymin


    x_cent = round(xmin + w/2)
    y_cent = round(ymin + h/2)
    x1 = x_cent - dot_width
    y1 = y_cent - dot_width
    x2 = x_cent + dot_width
    y2 = y_cent + dot_width

    # Draw the bounding box
    # ......................

    start_point = (x1, y1)
    end_point = (x2, y2)
    bbox_color = (255, 255, 255) # white
    bbox_thickness = line_thickness

    # Draw a filled square
    # -1 will fill the rectangle
    image = cv2.rectangle(image, start_point, end_point, bbox_color, -1)

    # Draw a border around the filled square in a different colour
    bbox_color = (255, 0, 255) # blue border
    image = cv2.rectangle(image, start_point, end_point, bbox_color, bbox_thickness)


    # Draw the background behind the text
    # ....................................

    # Only do this if text is not None.
    if text:
        # Draw the background behind the text
        #text_bground_color = (0, 0, 0)  # black
        #cv2.rectangle(image, (xmin, ymin - 150), (xmin + w, ymin), text_bground_color, -1)

        # Draw the text
        text_color = (255, 255, 255)  # white
        font = cv2.FONT_HERSHEY_DUPLEX
        origin = (xmin, ymin - 30)
        fontScale = 3
        thickness = 10

        image = cv2.putText(image, text, origin, font,
                            fontScale, text_color, thickness, cv2.LINE_AA)

    return image



# Transfer the uploaded images to png_images_dir.
# This applies to all images, png and jpg.
def transfer_images_to_folder(image_file_list):

    # Create png_images_dir.
    # Remember here we are outside the yolov5 folder.
    if os.path.isdir('yolov5/png_images_dir') == False:
        png_images_dir = os.path.join('yolov5', 'png_images_dir')
        os.mkdir(png_images_dir)

    # Prepare the images

    image_size_list = []

    for i, fname in enumerate(image_file_list):

        # Load an image
        path = 'uploads/' + fname
        image = cv2.imread(path)

        height = image.shape[0]
        width = image.shape[1]

        image_size_list.append((height, width))

        # Remember: Don't resize the image. Yolo does the resizing internally.


        # Save the image in the folder
        # that we created.
        dst = os.path.join(png_images_dir, fname)
        #image.save(dst)
        cv2.imwrite(dst, image)

    return image_size_list



def predict_on_all_uploaded_images(model_list, image_list, image_size_list):

    """
    This function returns a dataframe that has the following columns:
    [xmin, ymin, xmax, ymax, conf-score, class, image_id, fname, orig_image_height, orig_image_width]
    """

    IMAGE_SIZE = 512


    print('Starting prediction...')

    print(os.listdir('yolov5/png_images_dir'))



    # Get the path to each trained model
    model_path_0 = f"yolov5/TRAINED_MODEL_FOLDER/{model_list[0]}"

    # Instantiate the model.
    # Fix for path error:
    # https://stackoverflow.com/questions/74328379/typeerror-custom-got-an-unexpected-keyword-argument-path-yolov7
    # change path to path_or_model
    model = torch.hub.load('yolov5/', 'custom',
                           source='local', path_or_model=model_path_0,
                           force_reload=True)


    for i, image_fname in enumerate(image_list):

        # image_list is the list of file names of the images that were uploaded
        # image_size_list has the same order as image_list
        # image_size_list: [(h,w), (h,w), ...]
        orig_image_h = image_size_list[i][0]
        orig_image_w = image_size_list[i][1]


        image_id = image_fname.split('.')[0]
        image_path = f'yolov5/png_images_dir/{image_fname}'

        results = model(image_path, size=IMAGE_SIZE)

        outputs = results.xyxy[0]

        # If no wheat heads were detected on the image then
        # the shape of the output will be (0, 6).
        if outputs.shape[0] == 0:

            # Create a dataframe
            # Use a number e.g. 0 and not a str.
            # A str changes the dtype of the column which
            # will cause errors later.
            empty_dict = {
                'xmin': [0],
                'ymin': [0],
                'xmax': [0],
                'ymax': [0],
                'conf-score': [0],
                'class': ['no_objects']
            }

            df1 = pd.DataFrame(empty_dict)

            # Add the image_id column
            df1['image_id'] = image_id

            df1['fname'] = image_fname
            df1['orig_image_height'] = orig_image_h
            df1['orig_image_width'] = orig_image_w

        else:
            # convert to numpy so we can stack the batches.
            outputs = outputs.cpu().detach().numpy()

            # Create a dataframe
            cols = ['xmin', 'ymin', 'xmax', 'ymax', 'conf-score', 'class']
            df1 = pd.DataFrame(outputs, columns=cols)

            # Add the image_id column
            df1['image_id'] = image_id

            df1['fname'] = image_fname
            df1['orig_image_height'] = orig_image_h
            df1['orig_image_width'] = orig_image_w

        # stack the preds from each batch
        if i == 0:
            df_fin = df1
        else:
            df_fin = pd.concat([df_fin, df1], axis=0)

    # Save df_fin as a csv file.
    # This csv file gets used in the process_torch_hub_predictions() function.
    # This csv file also gets used later in the replace_dot_with_bbox() function.
    path = 'df_fin_preds.csv'
    df_fin.to_csv(path, index=False)

    print('Prediction completed.')

    return df_fin



def process_torch_hub_predictions(image_list, ABS_PATH_TO_STATIC):

    """
    This function:
    - Draws the dots on each image
    - Returns a dictionary containing the number of dots on each image.
    """

    print('Checking dir...')
    #print(os.getcwd())

    # Create pred_images_dir.
    # Remember here we are NOT inside the yolov5 folder
    if os.path.isdir('yolov5/pred_images_dir') == False:
        pred_images_dir = 'yolov5/pred_images_dir'
        os.mkdir(pred_images_dir)

    # We wull store the num preds for each image
    # in a dict. The image file name is the key and
    # the num preds is the value.
    num_preds_dict = {}

    # Load the dataframe
    df_preds = pd.read_csv('df_fin_preds.csv')

    for image_fname in image_list:

        image_id = image_fname.split('.')[0]

        df_test_preds = df_preds[df_preds['image_id'] == image_id]
        df_test_preds = df_test_preds.reset_index(drop=True)

        # Read the image
        path = os.path.join('yolov5/png_images_dir', image_fname)
        image = cv2.imread(path)


        # If no objects were detected on the image
        if df_test_preds.loc[0, 'class'] != 'no_objects':

            # Get the number of predicted bboxes
            num_bboxes = len(df_test_preds)

            # Add a key value pair to the dict
            num_preds_dict[image_fname] = num_bboxes

            # Draw the bboxes on the image
            for i in range(0, len(df_test_preds)):

                xmin = int(df_test_preds.loc[i, 'xmin'])
                ymin = int(df_test_preds.loc[i, 'ymin'])
                xmax = int(df_test_preds.loc[i, 'xmax'])
                ymax = int(df_test_preds.loc[i, 'ymax'])

                target = df_test_preds.loc[i, 'class']

                target_text = 'None'

                # Convert the target into a class name
                if target == 1:
                    target_text = 'head'

                # image = draw_bbox(image, xmin, ymin, xmax, ymax, text=None, line_thickness=5)
                image = draw_square_dot(image, xmin, ymin, xmax, ymax, 10, text=None, line_thickness=5)

            # save the image
            dst = os.path.join(pred_images_dir, image_fname)
            cv2.imwrite(dst, image)


         # If the image did not contain any objects
        else:

            # Add a key value pair to the dict
            num_preds_dict[image_fname] = 0

            # Read the image
            path = os.path.join('yolov5/png_images_dir', image_fname)
            image = cv2.imread(path)

            # save the image
            dst = os.path.join(pred_images_dir, image_fname)
            cv2.imwrite(dst, image)


    # Copy the pred_images_dir to the static folder so we can
    # display the images easily.

    # This is dependent on the current working directory.
    abs_path_to_pred_images_dir = os.path.abspath("yolov5/pred_images_dir")

    src = abs_path_to_pred_images_dir
    dst = os.path.join(ABS_PATH_TO_STATIC, "pred_images_dir")

    shutil.copytree(src, dst)


    return num_preds_dict



def delete_user_submitted_data():

    """
    Note:
    This function does not delete the images in 'static/pred_images_dir'.
    The app needs the png images in this folder to display them on the main page.
    The 'static/pred_images_dir' folder gets deleted each time the user submits new files and
    when the page first loads.

    """

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



# When the user clicks on a dot, this function
# replaces that dot with a bounding box.
def replace_dot_with_bbox(image_fname, dot_width=10):


    # Delete the analysis images folder if it exists.
    if os.path.isdir('static/analysis_images_dir') == True:
        shutil.rmtree('static/analysis_images_dir')
        print('Folder deleted.')

    # Create analysis_images_dir.
    if os.path.isdir('static/analysis_images_dir') == False:
        analysis_images_dir = 'static/analysis_images_dir'
        os.mkdir(analysis_images_dir)

    # Load the preds
    path = 'df_fin_preds.csv'
    df_preds1 = pd.read_csv(path)


    df = df_preds1[df_preds1['fname'] == image_fname]
    df = df.reset_index(drop=True)


    # Get the value of the 'pos_x' key
    pos_x = int(request.form.get('pos_x'))

    # Get the value of the 'pos_y' key
    pos_y = int(request.form.get('pos_y'))

    # Get the height and width of the displayed image
    image_display_h = int(request.form.get('image_display_h'))
    image_display_w = int(request.form.get('image_display_w'))

    pos_x_rel = pos_x / image_display_w
    pos_y_rel = pos_y / image_display_h


    # Load a fresh image without dots or bboxes
    path = os.path.join('yolov5/png_images_dir', image_fname)
    image = cv2.imread(path)


    # This will remain None if the user did not click on a dot
    new_image_str = 'None'


    # Identify that object with a bbox and the other objects with a dot
    # Draw the bboxes on the image
    for i in range(0, len(df)):

        # These values are based on the original image size.
        # If the image does not contain any objects then all
        # these values will be 0.
        xmin = int(df.loc[i, 'xmin'])
        ymin = int(df.loc[i, 'ymin'])
        xmax = int(df.loc[i, 'xmax'])
        ymax = int(df.loc[i, 'ymax'])


        orig_image_height = int(df.loc[i, 'orig_image_height'])
        orig_image_width = int(df.loc[i, 'orig_image_width'])

        w = xmax - xmin
        h = ymax - ymin

        x_cent = round(xmin + w / 2)
        y_cent = round(ymin + h / 2)
        x1 = x_cent - dot_width
        y1 = y_cent - dot_width
        x2 = x_cent + dot_width
        y2 = y_cent + dot_width

        pos_xx = pos_x_rel * orig_image_width
        pos_yy = pos_y_rel * orig_image_height


        if (pos_xx >= x1) and (pos_yy >= y1) and (pos_xx <= x2) and (pos_yy <= y2):

            # We will use this to create a new folder name.
            k = str(i)

            image = draw_bbox(image, xmin, ymin, xmax, ymax, text=None, line_thickness=5)

            # The problem:
            # We want to display the same image each time with just the bbox drawn in a different place.
            # But in the new_image_str code below the browser will not change the displayed image
            # if the src path to the new image is the same as for the previous image.
            # Solution:
            # We will store each changed image in a different folder. This will change the src path while
            # still keeping the same image_fname. We need the fname to stay the same because each time
            # we need to load the image that the user originally submitted, which is stored in png_images_dir.

            # We can change the folder name each time because
            # we only need the file name that's at the end of the src attribute.
            # As long as the file name stays the same each time everything will work.
            new_image_str = f"""<img id="selected-image" onclick="get_click_coords(event, this.src)"  class="w3-round unblock" src="/static/analysis_images_dir/{k}/{image_fname}"  height="580" alt="Wheat">"""


        else:

            image = draw_square_dot(image, xmin, ymin, xmax, ymax, 10, text=None, line_thickness=5)



    # Only if the user clicked on a dot
    if new_image_str != 'None':

        print('User clicked on a dot.')

        # Create analysis_images_dir.
        if os.path.isdir(f'static/analysis_images_dir/{k}') == False:
            analysis_images_dir = f'static/analysis_images_dir/{k}'
            os.mkdir(analysis_images_dir)

        # save the image
        dst = os.path.join(f'static/analysis_images_dir/{k}', image_fname)
        cv2.imwrite(dst, image)

    else:
        print('User did not click on a dot.')



    # If the user did not click on a dot then
    # new_image_str == 'None'.
    # Then the javascript code won't change the image on the page.
    # The existing image will remain as is.
    output = {
        'new_image_str': new_image_str,
        'pos_x': pos_x,
              'pos_y': pos_y,
              'image_display_h': image_display_h,
              'image_display_w': image_display_w
              }

    return output

