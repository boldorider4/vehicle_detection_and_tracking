import sys
import csv
import glob
import pickle
import zipfile
import tarfile
import os.path
import urllib.request
from sklearn.utils import shuffle as Shuffle
from features_extract import extract_features
from parameters import *


def load_and_extract():
  print('Downloading training/validation set from various sources...')

  # Read in car images
  cars = []
  dir_vehicles = './vehicles'
  if not os.path.isdir(dir_vehicles):
    zip_vehicles = './vehicles.zip'
    if not os.path.isfile(zip_vehicles):
      file_url = 'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip'
      print('Downloading ' + file_url + '...')
      urllib.request.urlretrieve(file_url, zip_vehicles)
    print('Unpacking ' + zip_vehicles + '...')
    zip_handle = zipfile.ZipFile(zip_vehicles, 'r')
    zip_handle.extractall('./')
    zip_handle.close()

  images = glob.glob(dir_vehicles + '/*/*.png')
  for image in images:
    cars.append(image)
  Shuffle(cars)

  # Read in non-car images
  notcars = []
  dir_vehicles = './non-vehicles'
  if not os.path.isdir(dir_vehicles):
    zip_vehicles = './non-vehicles.zip'
    if not os.path.isfile(zip_vehicles):
      file_url = 'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip'
      print('Downloading ' + file_url + '...')
      urllib.request.urlretrieve(file_url, zip_vehicles)
    print('Unpacking ' + zip_vehicles + '...')
    zip_handle = zipfile.ZipFile(zip_vehicles, 'r')
    zip_handle.extractall('./')
    zip_handle.close()

  images = glob.glob(dir_vehicles + '/*/*.png')
  for image in images:
    notcars.append(image)
  Shuffle(notcars)

  # Read in udacity car images
  # udacity_cars = []
  # dir_vehicles = './object-detection-crowdai'
  # if not os.path.isdir(dir_vehicles):
  #   tar_vehicles = './object-detection-crowdai.tar.gz'
  #   if not os.path.isfile(tar_vehicles):
  #     file_url = 'http://bit.ly/udacity-annoations-crowdai'
  #     print('Downloading ' + file_url + '...')
  #     urllib.request.urlretrieve(file_url, tar_vehicles)
  #   print('Extracting ' + tar_vehicles + '...')
  #   tar_handle = tarfile.open(name=tar_vehicles, mode='r')
  #   tar_handle.extractall('./')
  #   tar_handle.close()

  # with open(dir_vehicles + '/labels.csv') as csvfile:
  #     reader = csv.reader(csvfile)
  #     for line in reader:
  #         if 'Car' in line[5]:
  #             car_dict = { "file": line[4], "bbox": ((int(line[0]),int(line[1])), (int(line[2]),int(line[3]))) }
  #             udacity_cars.append(car_dict)
  # Shuffle(udacity_cars)

  print('Extracting features from vehicle/non-vehicle training/validation sets...')
  # Extract car features
  car_features = extract_features(cars, w_bboxes=False, color_space=color_space,
                                  spatial_size=spatial_size, hist_bins=hist_bins,
                                  orient=orient, pix_per_cell=pix_per_cell,
                                  cell_per_block=cell_per_block,
                                  hog_channel=hog_channel, spatial_feat=extract_spatial_features,
                                  hist_feat=extract_hist_bins_features, hog_feat=extract_hog_features)
  # Extract non-car features
  notcar_features = extract_features(notcars, w_bboxes=False, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=extract_spatial_features,
                                     hist_feat=extract_hist_bins_features, hog_feat=extract_hog_features)
  # Extract udacity car features
  udacity_cars_features = []
  #udacity_cars_features = extract_features(udacity_cars, w_bboxes=True, color_space=color_space,
  #                        spatial_size=spatial_size, hist_bins=hist_bins,
  #                        orient=orient, pix_per_cell=pix_per_cell,
  #                        cell_per_block=cell_per_block,
  #                        hog_channel=hog_channel, spatial_feat=extract_spatial_features,
  #                        hist_feat=extract_hist_bins_features, hog_feat=extract_hog_features)

  # Saving classifier to pickle file
  pickle_features = { "car_features" : car_features, "notcar_features" : notcar_features, \
                      "udacity_cars_features" : udacity_cars_features }
  with open('features.' + color_space.lower() + '.p', 'wb') as pf:
    pickle.dump(pickle_features, pf)
    print('features saved successfully!')
    pf.close()

if __name__ == '__main__':
  load_and_extract()
  sys.exit()
