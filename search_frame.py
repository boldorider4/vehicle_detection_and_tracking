from scipy.ndimage.measurements import label
from collections import deque
from features_extract import *
from parameters import *
from train_clf import *
import numpy as np
import pickle
import os


def search_window(img, ystart, ystop, scale, clf, X_scaler, window, orient, pix_per_cell, \
                  cell_per_block, hog_channel, color_space, spatial_size, hist_bins, \
                  spatial_feat=True, hist_feat=True, hog_feat=True, output_img=False):

    Debug_search_window = False
    draw_img = np.copy(img)
    if (np.max(img) > 1.0):
        img = img.astype(np.float32)/255
    else:
        img = img.astype(np.float32)

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps.
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    if Debug_search_window:
        print("\n\nscale {}".format(scale))
        print("ch1.shape {}".format(ch1.shape))

    # window steps
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    # Instead of overlap, define how many cells to step
    cells_per_step_x = 2
    cells_per_step_y = 2
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step_x + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step_y + 1
    if Debug_search_window:
        print("nysteps {}".format(nysteps))

    bboxes = []

    # Compute individual channel HOG features for the entire image
    if hog_feat:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step_x
                xpos = xb*cells_per_step_y

                # Extract HOG for this patch
                if hog_channel is 0:
                    hog_features = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                elif hog_channel is 1:
                    hog_features = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                elif hog_channel is 2:
                    hog_features = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                elif hog_channel is "ALL":
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                if spatial_feat:
                    spatial_features = bin_spatial(subimg, size=spatial_size)
                if hist_feat:
                    hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                feature_vec = []
                if spatial_feat:
                    feature_vec.append(spatial_features)
                if hist_feat:
                    feature_vec.append(hist_features)
                if hog_feat:
                    feature_vec.append(hog_features)

                features = np.concatenate(feature_vec).astype(np.float64)
                if Debug_search_window:
                    print("hog_features {}".format(len(hog_features)))
                    print("hist_features {}".format(len(hist_features)))
                    print("spatial_features {}".format(len(spatial_features)))
                test_features = X_scaler.transform(features.reshape(1, -1))
                test_prediction = clf.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    x1_y1 = (xbox_left, ytop_draw+ystart)
                    x2_y2 = (xbox_left+win_draw,ytop_draw+win_draw+ystart)
                    bboxes.append((x1_y1, x2_y2))
                    if output_img:
                        cv2.rectangle(draw_img, x1_y1, x2_y2, (0,0,255), 6)

    if output_img:
        return bboxes, draw_img
    else:
        return bboxes


def search_roi(input_image, y_array, scaling_factors, clf, X_scaler, win_size, orient, \
                                                              pix_per_cell, cell_per_block, \
                                                              hog_channel, color_space, spatial_size, hist_bins, \
                                                              spatial_feat, hist_feat, hog_feat, output_img=False):
    bboxes_list = []
    assert(len(y_array)==len(scaling_factors))

    for y_start_stop, scale in zip(y_array, scaling_factors):

        if output_img:
            bboxes, input_image = search_window(input_image, y_start_stop[0], y_start_stop[1], scale, \
                                             clf, X_scaler, win_size, orient, pix_per_cell, cell_per_block, \
                                             hog_channel, color_space, spatial_size, hist_bins, \
                                             spatial_feat, hist_feat, hog_feat, output_img)
        else:
            bboxes = search_window(input_image, y_start_stop[0], y_start_stop[1], scale, \
                                             clf, X_scaler, win_size, orient, pix_per_cell, cell_per_block, \
                                             hog_channel, color_space, spatial_size, hist_bins, \
                                             spatial_feat, hist_feat, hog_feat, output_img)
        bboxes_list.extend(bboxes)

    if output_img:
        return bboxes_list, input_image
    else:
        return bboxes_list


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    # Return the image
    return bboxes


# global bounding box history
boxhistory = deque([])
boxhistory_length = 10
# if not already done, train classifier
if not os.path.isfile('clf_scaler.' + color_space.lower() + '.p'):
  train_classifier()

# Restore previously saved classifier and scaler
if os.path.isfile('clf_scaler.' + color_space.lower() + '.p'):
  with open('clf_scaler.' + color_space.lower() + '.p', 'rb') as pf:
    pickle_clf = pickle.load(pf)
    clf = pickle_clf["clf"]
    X_scaler = pickle_clf["X_scaler"]
    print('classifier restored successfully!')
    pf.close()

def detection_pipeline(input_image):

    global scaling_factors, clf, X_scaler, win_size, orient, pix_per_cell, cell_per_block, \
           hog_channel, spatial_size, hist_bins, extract_spatial_features, extract_hist_bins_features, \
           extract_hog_features, boxhistory, boxhistory_length, Debug, y_array

    # generate empty heat map
    heat = np.zeros_like(input_image[:,:,0]).astype(np.float)

    # find bounding boxes
    bboxes = search_roi(input_image, y_array, scaling_factors, clf, X_scaler, win_size, orient, \
                        pix_per_cell, cell_per_block, hog_channel, color_space, spatial_size, hist_bins, \
                        spatial_feat=extract_spatial_features, hist_feat=extract_hist_bins_features, \
                        hog_feat=extract_hog_features, output_img=False)

    # Add heat to each box in box list
    heat = add_heat(heat, bboxes)

    # debug dict for storing intermediate pipeline stages
    returned_debug = {}

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)
    if Debug:
        returned_debug["heat"] = np.copy(heat)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    if Debug:
        returned_debug["labels"] = np.copy(labels)
    bboxes_labeled = draw_labeled_bboxes(np.copy(input_image), labels)
    debug_drawn_boxes = np.copy(input_image)
    for bbox in bboxes_labeled:
        cv2.rectangle(debug_drawn_boxes, bbox[0], bbox[1], (0,0,255), 6)
    if Debug:
        returned_debug["debug_drawn_boxes"] = debug_drawn_boxes

    if len(boxhistory) >= boxhistory_length:
        boxhistory.popleft()
    boxhistory.append(bboxes_labeled)

    heat = np.zeros_like(input_image[:,:,0]).astype(np.float)
    for bboxes in boxhistory:
        heat = add_heat(heat, bboxes)

    heat = apply_threshold(heat,2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    bboxes_labeled = draw_labeled_bboxes(np.copy(input_image), labels)

    # Draw the box on the image
    output_img = np.copy(input_image)
    for bbox in bboxes_labeled:
        cv2.rectangle(output_img, bbox[0], bbox[1], (0,0,255), 6)

    if Debug:
        return output_img, returned_debug
    else:
        return output_img
