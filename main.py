import cv2
import os
import numpy as np
import csv

master_path_to_dataset = "TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed
dir_path = os.path.dirname(os.path.realpath(__file__))


# -------------------------- SKIP FORWARD TO A SPECIFIC TIMESTAMP --------------------------
# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns
skip_forward_file_pattern = ""  # set to timestamp to skip forward to
# ------------------------------------------------------------------------------------------

# ------------------ YOLO CONFIG --------------------
f = open("coco.names", "r", encoding='mbcs')
classes = f.read().rstrip('\n').split('\n')
# ---------------------------------------------------

#####################################################################
# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in


def drawPred(image, class_name, confidence, left, top, right, bottom, colour):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 6)

    # construct label
    label = '%s:%.2f' % (class_name, confidence)

    #Display the label at the top of the bounding box
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

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

################################################################################

# init YOLO CNN object detection model

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image


# load configuration and weight files for the model and load the network using them

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
output_layer_names = getOutputsNames(net)

 # defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

################################################################################

# dummy on trackbar callback function
def on_trackbar(val):
    return

# define display window name + trackbar

windowName = 'YOLOv3 object detection: ' + "yolov3.weights"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
trackbarName = 'reporting confidence > (x 0.01)'
cv2.createTrackbar(trackbarName, windowName , 0, 100, on_trackbar)

################################################################################



# ------------------------- CALIBRATION ----------------------------------
# fixed camera parameters for this stereo setup (from calibration)
camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000  # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502  # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5
# ----------------------------------------------------------------------

crop_disparity = True  # display full or cropped disparity image
pause_playback = False  # pause until key press after each image


def project_disparity_to_3d(disparity, rgb=None):
    # RGB is an image with default value []
    points = []
    f = camera_focal_length_px
    B = stereo_camera_baseline_m
    height, width = disparity.shape[:2]

    # assume a minimal disparity of 2 pixels is possible to get Zmax and then we get reasonable
    # scaling in X and Y output if we change Z to Zmax in the lines X = ....; Y = ...; below
    # Zmax = ((f * B) / 2);

    # Cycles through every point in the image
    for y in range(height):  # 0 - height is the y axis index
        for x in range(width):  # 0 - width is the x axis index

            # if we have a valid non-zero disparity
            if disparity[y, x] > 0:
                # calculate corresponding 3D point [X, Y, Z] i.e. stereo lecture - slide 22 + 25
                Z = (f * B) / disparity[y, x]
                X = ((x - image_centre_w) * Z) / f
                Y = ((y - image_centre_h) * Z) / f

                # add to points
                if rgb.size:
                    points.append([X, Y, Z, rgb[y, x, 2], rgb[y, x, 1], rgb[y, x, 0]])
                else:
                    points.append([X, Y, Z])
    return points


# resolve full directory location of data set for left / right images
full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

# get a list of the left image files and sort them (by timestamp in filename)
left_file_list = sorted(os.listdir(full_path_directory_left))

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)
# uses a modified H. Hirschmuller algorithm [Hirschmuller, 2008] that differs (see opencv manual)
# parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]

# FROM manual: stereoProcessor = cv2.StereoSGBM(numDisparities=128, SADWindowSize=21);
# From help(cv2): StereoBM_create(...)
#        StereoBM_create([, numDisparities[, blockSize]]) -> retval
#
#    StereoSGBM_create(...)
#        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
# disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval

max_disparity = 128
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
    print(full_path_filename_left)
    print(full_path_filename_right)
    print()

    # check the file is a PNG file (left) and check a corresponding right image actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):
        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        # cv2.imshow('left image', imgL)

        # Got left image, now YOLO time
        frame = imgL
        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # set the input to the CNN network
        net.setInput(tensor)
        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        confThreshold = cv2.getTrackbarPos(trackbarName, windowName) / 100
        classIDs, confidences, boxes = postprocess(frame, results, confThreshold, nmsThreshold)

        # draw resulting detections on image
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(frame, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width, top + height, (255, 0, 0))

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        fullscreen = False

        # display image
        cv2.imshow(windowName, frame)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN & fullscreen)


        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        # cv2.imshow('right image', imgR)

        print("-- files loaded successfully\n")

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

        # ----------------- DISPARITY CALCULATIONS FINISHED HERE -----------------------
        # project to a 3D colour point cloud (with or without colour)

        # Without color
        # points = project_disparity_to_3d(disparity_scaled);

        # With color
        points = project_disparity_to_3d(disparity_scaled, imgL)

        # write to file in an X simple ASCII X Y Z format that can be viewed in 3D
        # using the on-line viewer at http://lidarview.com/
        # (by uploading, selecting X Y Z format, press render , rotating the view)

        point_cloud_file = open('3d_points.txt', 'w')
        csv_writer = csv.writer(point_cloud_file, delimiter=' ')
        csv_writer.writerows(points)
        point_cloud_file.close()
        # cv2.imshow("disparity", (disparity_scaled * (256. / max_disparity)).astype(np.uint8))

        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # crop - c
        # pause - space

        key = cv2.waitKey(40 * (not pause_playback)) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if key == ord('x'):       # exit
            break  # exit
        elif key == ord('s'):     # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled)
            cv2.imwrite("left.png", imgL)
            cv2.imwrite("right.png", imgR)
        elif key == ord('c'):     # crop
            crop_disparity = not crop_disparity
        elif key == ord(' '):     # pause (on next frame)
            pause_playback = not pause_playback
    else:
        print("-- files skipped (perhaps one is missing or not PNG)\n")

# close all windows
cv2.destroyAllWindows()
