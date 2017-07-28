from moviepy.editor import VideoFileClip
from IPython.display import HTML
from search_frame import *
from train_clf import *
from parameters import *
import matplotlib.pyplot as plt


def main():
  if Debug_short_video:
    video_output_file = 'project_video_out_debug.mp4'
  else:
    video_output_file = 'project_video_out.mp4'

  video_input = VideoFileClip("./project_video.mp4")
  if Debug_short_video:
    video_input = video_input.subclip(6, 14)

  # For debugging purposes, some stages of the pipeline stages can be outputted while
  # running the lane finding on the video frames
  if not Debug:
    video_output = video_input.fl_image(detection_pipeline)
    video_output.write_videofile(video_output_file, audio=False)
  else:
    for t in np.linspace(30, 34, 10):
      image = video_input.get_frame(t)
      print('\ntime {}'.format(t))
      found_cars = detection_pipeline(image)
      #found_cars = image
      plt.figure(figsize=(15,15))
      plt.imshow(found_cars)


if __name__ == '__main__':
  main()
  sys.exit()
