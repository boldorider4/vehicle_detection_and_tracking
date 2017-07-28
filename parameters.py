color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
extract_spatial_features = True # Spatial features on or off
extract_hist_bins_features = True # Histogram features on or off
extract_hog_features = True # HOG features on or off

y_array = ((400, 480), (400, 600), (400, 630), (450, 720))
# different scaling steps applied along y axis
scaling_factors = (1.1, 1.5, 2.0, 2.5)
win_size = 64

Debug = False
Debug_short_video = False
