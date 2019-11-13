import cv2
import os
import numpy as np

# Dataset path configuration
master_path_to_dataset = "TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

# resolve full directory location of data set for left / right images
full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

# get a list of the left image files and sort them (by timestamp in filename)
left_file_list = sorted(os.listdir(full_path_directory_left))

# Skip forward to specific timestamp e.g. set to 1506943191.487683
skip_forward_file_pattern = ""

# ------------------------- CALIBRATION ----------------------------------
# fixed camera parameters for this stereo setup (from calibration)
camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000  # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502  # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5
# ----------------------------------------------------------------------

# Disparity
crop_disparity = True # display full or cropped disparity image
max_disparity = 128

# YOLO configuration
f = open("coco.names", "r", encoding='mbcs')
classes = f.read().rstrip('\n').split('\n')

################################################################################
# init YOLO CNN object detection model
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

# load configuration and weight files for the model and load the network using them
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# Get the names of all the layers in the network
layersNames = net.getLayerNames()
output_layer_names = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

 # defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
################################################################################



#####################################################################
# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in


def drawPred(image, class_name, confidence, box, z_mode):
    # choose colour depending on distance away
    tone = 255 - (min(z_mode, 50)/50)*255
    colour = (tone*0.90, tone*0.85, tone*0.678)

    # Draw a bounding box.

    left, top, width, height = box
    cv2.rectangle(image, (left, top), (left + width, top + height), colour, 2)

    label = '%s:%.2fm' % (class_name, z_mode)     # construct label

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)


#####################################################################
# Remove the bounding boxes with low confidence using non-maximal suppression
# image: image detection performed on
# results: output from YOLO CNN network
# threshold_confidence: threshold on keeping detection
# threshold_nms: threshold used in non maximum suppression
def postprocess(image, results, threshold_confidence, threshold_nms):
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
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)
################################################################################
# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object


# define display window name + trackbar
windowName = 'Stereo Vision with Distance Ranging'
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
trackbarName = 'reporting confidence > (x 0.01)'
cv2.createTrackbar(trackbarName, windowName, 0, 100, lambda: None)

################################################################################

def project_point_to_3D(img_disparity, x, y):
    # RGB is an image with default value []
    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    # assume a minimal disparity of 2 pixels is possible to get Zmax and then we get reasonable
    # scaling in X and Y output if we change Z to Zmax in the lines X = ....; Y = ...; below
    # Zmax = ((f * B) / 2);
    X = Y = Z = None
    max_height, max_width = img_disparity.shape[:2]
    x = min(max_width-1, x)
    y = min(max_height-1, y)

    # if we have a valid non-zero disparity
    if img_disparity[y, x] > 0:
        # calculate corresponding 3D point [X, Y, Z] i.e. stereo lecture - slide 22 + 25
        Z = (f * B) / img_disparity[y, x]
        X = ((x - image_centre_w) * Z) / f
        Y = ((y - image_centre_h) * Z) / f
    return X, Y, Z


def object_depth(img_disparity, box_size):
    # get the mode depth of object
    left_point, top_point, box_width, box_height = box_size
    z_values = []
    for y in range(top_point, top_point + box_height - 1):
        for x in range(left_point, left_point + box_width - 1):
            _, _, z = project_point_to_3D(img_disparity, x, y)
            if z:
                z_values.append(z)
    if not z_values:
        print(box)
        print(img_disparity[0:box_width])
    print(np.median(z_values))
    return np.median(z_values)


# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

# OPEN YOLO WINDOW # ---------------------------------
# create window by name (as resizable)
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)


# Cycles through all files, gets the corresponding right image and gets the full paths of both of them
for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)
    if (len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left):
        continue
    elif (len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left):
        skip_forward_file_pattern = ""

    # from the left image filename get the corresponding right image
    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

    # for sanity print out these filenames
    print(full_path_filename_left + '\n' + full_path_filename_right + '\n')

    # check the file is a PNG file (left) and check a corresponding right image actually exists
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):
        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel RGB images so load both as such
        print("-- files loaded successfully\n")


        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # perform pre-processing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation

        grayL = np.power(grayL, 0.75).astype('uint8')
        grayR = np.power(grayR, 0.75).astype('uint8')

        # -------------------------- DISPARITY ------------------------
        # compute disparity image from undistorted and rectified stereo images
        # that we have loaded
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)
        disparity = stereoProcessor.compute(grayL, grayR)

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

        if crop_disparity:
            width = np.size(disparity_scaled, 1)
            disparity_scaled = disparity_scaled[0:390, 135:width]

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)
        # --------------------------------------------------------------------------------

        # Got left image, now YOLO time
        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)

        tensor = cv2.dnn.blobFromImage(imgL, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # set the input to the CNN network
        net.setInput(tensor)
        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        confThreshold = cv2.getTrackbarPos(trackbarName, windowName) / 100
        classIDs, confidences, boxes = postprocess(imgL, results, confThreshold, nmsThreshold)

        # draw resulting detections on image
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            z_mode = object_depth(disparity_scaled, box)
            if not np.isnan(z_mode):
                drawPred(imgL, classes[classIDs[detected_object]], confidences[detected_object], box, z_mode)
        fullscreen = False

        # display image
        cv2.imshow(windowName, imgL)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN & fullscreen)

        key = cv2.waitKey(100 * 1) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

    else:
        print("-- files skipped (perhaps one is missing or not PNG)\n")

# close all windows
cv2.destroyAllWindows()
