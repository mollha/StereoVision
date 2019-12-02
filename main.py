################################################################################################################
# ATTRIBUTION - significant portions of this project are based upon scripts written by Toby Breckon
# - modified script for implementing YOLO object detection
# - modified Toby's script for loading computing SGBM disparity
# - modified SGBM disparity to 3D points for am example pair

# Copyright (c) 2019 Toby Breckon, Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html
# Author : Toby Breckon, toby.breckon@durham.ac.uk
# Implements the You Only Look Once (YOLO) object detection architecture described in full in:
# Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv pre-print arXiv:1804.02767.
# https://pjreddie.com/media/files/papers/YOLOv3.pdf
################################################################################################################
import math

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------- Configure paths to Data Set ----------------------------------------
master_path_to_dataset = "TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"
full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)
skip_forward_file_pattern = "1506942658.476606"  # Skip forward to specific timestamp e.g. set to 1506943191.487683

# Get a list of the left image files and sort them (by timestamp in filename)
left_file_list = sorted(os.listdir(full_path_directory_left))

# ------------------------------------------- Toggle Settings -----------------------------------------------
enable_keys = True  # enable the use of save "s" and exit "x" keys
fullscreen = True  # set the left colour image with bounding boxes to fullscreen
crop_disparity = True  # display full or cropped disparity image
max_disparity = 128  # set the maximum

# -------------------------------------------- Stereo Calibration --------------------------------------------
# fixed camera parameters for this stereo setup (from calibration)
camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000  # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502  # camera baseline in metres
image_centre_h, image_centre_w = 262.0, 474.5

# ----------------------------------------- YOLO Configuration ------------------------------------------------
f = open("coco.names", "r", encoding='mbcs')
classes = f.read().rstrip('\n').split('\n')

# init YOLO CNN object detection model
confThreshold, nmsThreshold = 0.5, 0.4  # Confidence threshold and Non-maximum suppression threshold
inpWidth, inpHeight = 416, 416  # Width of network's input image and Height of network's input image

# load configuration and weight files for the model and load the network using them
# net : an OpenCV DNN module network object
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

layersNames = net.getLayerNames()  # Get the names of all the layers in the network
# Get the names of the output layers of the CNN network
output_layer_names = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
###############################################################################################################

# HELPER FUNCTIONS FOR IMPROVING DISTANCE RANGING AND / OR OBJECT DETECTION

def k_means_depth(disparity_map, box_size):
    f = camera_focal_length_px
    B = stereo_camera_baseline_m
    left_point, top_point, box_width, box_height = box_size  # unpack box size
    max_height, max_width = disparity_map.shape[:2]
    box_width = min(left_point + box_width, max_width - 1) - left_point
    box_height = min(top_point + box_height, max_height - 1) - top_point
    cropped_map = disparity_map[top_point: top_point + box_height, left_point: left_point + box_width].copy()
    print(cropped_map.shape)
    # cv2.imwrite('cropmapp_unmeans.png', cropped_map)
    # Use k-means to separate an object region into the foreground and background
    z = cropped_map.reshape((-1, 1))
    print(z.shape)
    z = np.float32(z)  # convert to np.float32

    # define criteria, number of clusters(K) and apply k_means()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2
    # set random INITIAL centers
    ret, x_label, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centroids = center
    center = np.uint8(center)
    print(center)
    response = center[x_label.flatten()].reshape(cropped_map.shape)
    # plt.hist(response, 128, facecolor='red', alpha=0.5)
    # plt.show()
    # cv2.imshow('cropped', response)
    # cv2.imwrite('cropm.png', response)

    # Assumes the object
    max_center = float('-inf')
    for center_list in centroids:
        if max(center_list) > max_center:
            max_center = max(center_list)
    print(max_center)
    return (f * B) / max_center


def draw_pred(image: np.ndarray, class_name: str, box_dimensions: list, distance: float) -> None:
    """
    Draw the predicted bounding box on the specified image (mainly the left colour image)
    :param image: image upon which object detection was performed
    :param class_name: string name of detected object_detection
    :param box_dimensions: rectangle parameters for detection
    :param distance: depth of object (included in label)
    :return: NoneType (no return value)
    """

    # Params describes the amount of scaling in each of the rgb channels to produce a different color for each class
    # The main object classes have different colours to make them easier to distinguish
    params = {"car": (1, 0.85, 0.678), "person": (0, 0, 1), "truck": (0.33, 1, 0.5), "bus": (0.506, 0.149, 0.965)}
    if class_name in params:
        colour_scale = params[class_name]  # get the colour values from the colour dictionary
    else:
        colour_scale = (0.90, 0.85, 0.678)  # standard colour for less frequently occurring objects

    # Choose the brightness of the colour depending on distance away
    # Objects that are further away have a darker shade of their class colour
    # This gets brighter as the object gets closer - enables us to better visualise the depth predictions
    tone = 255 - (min(distance, 50)/50)*255
    colour = (tone * colour_scale[0], tone * colour_scale[1], tone * colour_scale[2])

    # Draw a bounding box.
    left, top, box_width, height = box_dimensions  # unpack the box tuple
    cv2.rectangle(image, (left, top), (left + box_width, top + height), colour, 2)
    object_label = ' %s : %.2fm' % (class_name, distance)     # construct label

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(object_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
                  (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),
                  (0, 0, 0))
    cv2.putText(image, object_label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)


def postprocess(image, results, threshold_confidence, threshold_nms):
    """
    Remove the bounding boxes with low confidence using non-maximal suppression
    :param image: image detection performed on
    :param results: output from YOLO CNN network
    :param threshold_confidence: threshold on keeping detection
    :param threshold_nms: threshold used in non maximum suppression
    :return: list of tuples
    """
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = max(0, int(center_x - width / 2))
                top = max(0, int(center_y - height / 2))
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    list_of_tuples = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        list_of_tuples.append((classIds[i], confidences[i], boxes[i]))
    # return post processed lists of classIds, confidences and bounding boxes
    return list_of_tuples


def yolo_pre_filtering(left_img):
    # Use an image pre-filtering technique to improve detection
    # Apply the bilateral filter with optimal parameters (described in the report)
    return cv2.bilateralFilter(left_img, 5, 35, 160)


def image_pre_filtering(left_img, right_img):
    # Use an image pre-filtering technique to improve disparity calculation
    # Apply this pre-filtering to both the left and right images after their grayscale conversion

    # # APPLY CLAHE
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    def logarithmic(image):
        c = max_disparity / math.log(1 + np.max(image))
        sigma = 1
        for i in range(0, image.shape[1]):  # image width
            for j in range(0, image.shape[0]):  # image height
                # compute logarithmic transform
                image[j, i] = int(c * math.log(1 + ((math.exp(sigma) - 1) * image[j, i])))
        return image

    def exponential(image):
        # perform pre-processing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation
        return np.power(image, 0.75).astype('uint8')

    def apply_filter(image):
        # choose filters to apply
        return exponential(image)

    return apply_filter(left_img), apply_filter(right_img)


def disparity_post_processing(matcher, disparity_L, disparity_R, imgL):
    # Uses Weighted Least Squares filter
    # FILTER Parameters
    lmbda = 8000
    sigma = 0.8
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher)
    wls.setLambda(lmbda)
    wls.setSigmaColor(sigma)
    disparity_filtered = wls.filter(disparity_L, imgL, None, disparity_R)
    return disparity_filtered

def apply_heuristics(object_class: str, box_dimensions):
    # RULE to crop pedestrian and car object regions
    # Alter the box_dimensions in order to crop object regions based on class
    left_point, top_point, box_width, box_height = box_dimensions  # unpack box size

    # For vehicles, crop out the upper 50% - this should remove the window reflections
    if object_class in ['car', 'truck', 'bus']:
        box_height = int(box_height // 2)
        top_point = top_point + box_height
    elif object_class in ['person, bicycle', 'motorbike']:
        # For pedestrians and bicycles, crop out the ground beneath them - the lower 20%
        top_point = top_point + int(box_height // 5)
        box_height = int(box_height // 1.25)
    return [left_point, top_point, box_width, box_height]


def median_depth(img_disparity, box_size):
    # Compute the median depth in the object region
    f = camera_focal_length_px
    B = stereo_camera_baseline_m
    max_height, max_width = img_disparity.shape[:2]
    left_point, top_point, box_width, box_height = box_size  # unpack box size
    d_values = []  # create empty list which will contain depth values
    for y in range(top_point, top_point + box_height - 1):
        for x in range(left_point, left_point + box_width - 1):
            x = min(max_width - 1, x)
            y = min(max_height - 1, y)

            if img_disparity[y, x] > 0:
                d = (f * B) / img_disparity[y, x]
                if d:
                    d_values.append(d)
    return np.median(d_values)


# setup the disparity stereo processor to find a maximum of 256 disparity values
window_size = 5
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=max_disparity,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=5,
    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

################################################################################################################
# Cycles through all file pairs
for filename_left in left_file_list:

    # ------------------------------------ GET THE CORRECT IMAGE PAIR -------------------------------------
    # Skip forward to start from a file we specify by timestamp (if this is set)
    if (len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left):
        continue
    elif (len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left):
        skip_forward_file_pattern = ""

    # From the left image filename get the corresponding right image
    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

    # For sanity, print out these filenames
    print(full_path_filename_left + '\n' + full_path_filename_right + '\n')

    # check the file is a PNG file (left) and check a corresponding right image actually exists
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):
        # N.B. despite one being grayscale both are in fact stored as 3-channel RGB images so load both as such
        print("-- files loaded successfully\n")

        # Read in the image pair, N.B. despite one being grayscale both are in fact stored as 3-channel RGB images
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        # convert to grayscale (as the disparity matching works on grayscale) - need to do for both (3-channel images)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # apply image pre-filtering to the stereo pair, as described in the report
        filtered_imgL, filtered_imgR = image_pre_filtering(grayL, grayR)

        # -------------------------- DISPARITY ------------------------
        # compute disparity image from undistorted and rectified stereo images
        # that we have loaded
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)
        disparity_L = left_matcher.compute(grayL, grayR)
        disparity_R = right_matcher.compute(grayR, grayL)
        disparity = disparity_post_processing(left_matcher, disparity_L, disparity_R, filtered_imgL)
        # disparity = stereoProcessor.compute(grayL, grayR)


        # filter out noise and speckles (adjust parameters as needed)
        dispNoiseFilter = 5  # increase for more aggressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

        # scale the disparity to 8-bit for viewing
        # divide by 16 and convert to 8-bit image (then range of values should
        # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
        # so we fix this also using a initial threshold between 0 and max_disparity
        # as disparity=-1 means no disparity available

        _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (disparity / 16.).astype(np.uint8)

        # crop disparity to chop out left part where there is no disparity
        # as this area is not seen by both cameras and also
        # chop out the bottom area (where we see the front of car bonnet)

        # APPLY k-means

        # display image

        if crop_disparity:
            width = np.size(disparity_scaled, 1)
            disparity_scaled = disparity_scaled[0:390, 135:width]
            imgL = imgL[0:390, 135:width]

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)

        # --------------------- PERFORM YOLO OBJECT DETECTION AND YOLO INPUT PRE-FILTERING ----------------------------
        # Apply pre-filtering to the left colour image (input to YOLO) as described in the report
        # This enhances object detection
        pre_filtered_YOLO_img = yolo_pre_filtering(imgL)

        # Create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(pre_filtered_YOLO_img, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # Set the input to the CNN network
        net.setInput(tensor)
        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)
        # remove the bounding boxes with low confidence
        detected_objects = postprocess(imgL, results, confThreshold, nmsThreshold)

        # ------------------------- CALCULATE THE DEPTHS OF OBJECTS DETECTED IN THE SCENE -----------------------------
        processed_objects = []  # this list will contain tuples representing detected objects and their depths
        for detected_object in detected_objects:
            classID, confidence, box = detected_object  # unpack the detected object
            print(box)
            object_class = classes[int(classID)]  # get the class name from the class ID
            cropped_box = apply_heuristics(object_class, box)
            # produce a single representative value for the depth of the object
            distance_prediction = median_depth(disparity_scaled, cropped_box)  # produce a single depth value from an object
            distance_prediction = k_means_depth(disparity_scaled, cropped_box)

            if not np.isnan(distance_prediction):  # check that this value is not NaN
                distance_prediction -= 1.1  # subtract avg car bonnet length
                processed_objects.append((box, object_class, float(distance_prediction)))

        # ----------------------------- DRAW THE BOUNDING BOXES ON THE LEFT COLOUR IMAGE ------------------------------
        # Sort the detected objects by their depth in the scene, with the furthest objects at the beginning of the list
        sorted_objects = sorted(processed_objects, key=lambda x: x[2], reverse=True)
        # Draw the bounding boxes around detected objects, draw furthest objects first
        for detected_object in sorted_objects:
            # Draw the bounding boxes (predictions) on the left colour image
            draw_pred(imgL, detected_object[1], detected_object[0], detected_object[2])

        # ----------------------------------------- DISPLAY THE IMAGE WINDOWS -----------------------------------------
        windowName = 'Stereo Vision with Distance Ranging'
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.imshow(windowName, imgL)  # show the left colour image with the bounding boxes
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN & fullscreen)  # initialise window properties
        cv2.imshow("Disparity Map", (disparity_scaled * (256. / max_disparity)).astype(np.uint8))  # show the disparity

        # ------------------------------------- ENABLE THE EXIT AND SAVE KEYS -----------------------------------------
        key = cv2.waitKey(10 * 1) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if enable_keys:
            if key == ord('x'):  # if key == 'x', i.e. exit key is pressed
                break  # exit
            elif key == ord('s'):  # if key == 's', i.e. save key is pressed
                label = 'Timestamp : %s' % filename_left[0:-6]  # Create the label containing the timestamp
                # Draw the timestamp on the image
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(imgL, (15, 15 - round(1.2 * labelSize[1])),
                              (15 + round(1.2 * labelSize[0]), 15 + baseLine), (255, 255, 255), cv2.FILLED)
                cv2.putText(imgL, label, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                # Write the left colour image with the bounding boxes and timestamp to local filesystem
                cv2.imwrite(filename_left, imgL)
        # -------------------------------------------------------------------------------------------------------------
    else:
        print("-- files skipped (perhaps one is missing or not a PNG)\n")  # Alert problem with file(s)

# Close all windows
cv2.destroyAllWindows()
