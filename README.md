# Vehicle Detection and Tracking
A Python program to detect and track vehicles in videos taken from car dash cams. The program is based on the implementation of a image processing pipeline where a combination of Computer Vision and Machine Learning techniques are used to detect passing cars.

### Note
Before running the routine it is advised to have a proper Python environment set-up. To ensure reproduceability it is best to use an [Anaconda/Miniconda](https://www.continuum.io/downloads) environment.
The Anaconda environment used for development is available at this [repo](https://github.com/udacity/CarND-Term1-Starter-Kit). Follow the [configuration help](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md) to install the environment.

### Description of execution
The program downloads a training/validation set from specific sources, extract features from each data set, trains a classifier to tell car images from non-car images and analyzes an input video (by default the one committed to this project is called 'project_video.mp4') frame by frame using a sliding window technique. Ultimately, the trained classifier is used to predict whether the given window contains a full or partial picture of a car. Several heat-mapping and masking techniques are used to reduce false positives and track the detection of a vehicle across a series of frames.

### Description of execution
The output consists of a copy of the input project video with bounding boxes drawn around the detected vehicles.

### Usage
#### Optional
```
 $ source activate carnd-term1
```
#### Launch the main program
```
 $ python main.py
```
