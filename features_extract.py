from skimage.feature import hog
import matplotlib.image as mpimg
import numpy as np
import cv2


def convert_color(img, conv='RGB'):
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif conv is not 'RGB':
        print('Invalid color space')
    return img


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features(imgs, w_bboxes=False, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        if w_bboxes:

            bbox = file["bbox"]
            min_bbox_size = min( bbox[1][1]-bbox[0][1], bbox[1][0]-bbox[0][0] )
            max_bbox_size = max( bbox[1][1]-bbox[0][1], bbox[1][0]-bbox[0][0] )

            # check if bounding boxes indeces are ok
            if min_bbox_size is 0 or max_bbox_size is 0:
                continue
            # calculate bounding box aspect ratio
            bbox_aratio = max_bbox_size / min_bbox_size
            # if aspect ratio is not adequate or if bounding box is too small, skip image
            if min_bbox_size >= 64 and bbox_aratio < 2.5:
                image = mpimg.imread(os.path.join('.', 'object-detection-crowdai', file["file"]))
                image = image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
                # if bounding box is too far from square shape, crop square shape out of it
                if bbox_aratio > 1.2:
                    image = image[0:min_bbox_size, 0:min_bbox_size]

                # resize image to standard 64-by-64 image
                image_0 = cv2.resize(image[:,:,0], (64,64))
                image_1 = cv2.resize(image[:,:,1], (64,64))
                image_2 = cv2.resize(image[:,:,2], (64,64))
                image = np.dstack((image_0, image_1, image_2))
            else:
                continue
        else:
            image = mpimg.imread(file)

        # scaling to [0, 1] range in case range was initially [0, 255] to enforce consistency
        if (np.max(image) > 1.0):
            image = image.astype(np.float32)/255
        else:
            image = image.astype(np.float32)

        # apply color conversion if other than 'RGB'
        feature_image = convert_color(np.copy(image), conv=color_space)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
