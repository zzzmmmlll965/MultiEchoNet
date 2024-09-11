# MultiEchoNet
A multi-task network for simultaneous left ventricular segmentation and keypoint localization

a4c-video-dir: Datasets for training, validation, and testing
----videos——Processed AVI video files
----FileList.csv——Video information required for training and verification
----VolumeTracings.csv——Some of the information elements of the video
----keypoints.csv——Key point annotation information required for training and verification

1、The first thing you need to do is run setup.py files, install the necessary packages, and package echonet into a library
python setup.py install 

2、Install the required libraries


python setup.py install 

3、Modify echonet/utils/segmentation.py, line 21 --data_dir

4、run
python segmentation.py

After the run, the following file content will be generated:
log.csv: Metric records for training and validation
checkpoint.pt: Checkpoint
best.pt: The weight checkpoint for the model with the lowest validation loss
size.csv: Estimated size of the left ventricle per frame and an indicator of the start of the heartbeat
vedios: A directory that contains videos with segmented overlays

Some of the code is being updated continuously~~~
