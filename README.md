# Eyes-in-the-Sky-Object-Detection-with-a-Drone
"Eyes in the Sky: Object Detection with a Drone" using Pygame, DJITelloPy, OpenCV to control drone movement, detect objects, save images/videos. Built with Python.
Eyes in the Sky: Object Detection with a Drone
Welcome to the "Eyes in the Sky" project, a script that uses the power of Pygame and OpenCV to create a window for displaying live video feed from a drone controlled by the DJITelloPy library. The script uses OpenCV's dnn_DetectionModel to perform object detection on the video frames and display the results on the screen.

Getting Started
Before you begin, you will need to have the following software installed on your computer:

Python 3.x
Pygame
OpenCV
DJITelloPy
You will also need to have access to a DJI Tello drone.


Object Detection
The script uses OpenCV's dnn_DetectionModel to perform object detection on the video frames. The model is pre-trained on the COCO dataset, which means it can detect 80 different object classes, including people, cars, and animals. The results of the object detection are displayed on the screen as bounding boxes around the detected objects, along with the class label and confidence score.

You can adjust the confidence threshold for object detection by modifying the conf_threshold variable in the script. A lower threshold will detect more objects, but may also result in more false positives.

Conclusion
We hope you enjoy using this script to control your drone and detect objects in the video feed. With the help of this script, you can easily integrate object detection capabilities into your drone projects and explore new ways to use drones for object detection.

If you have any questions or issues with the script, please feel free to open an issue on GitHub or contact us directly. We are happy to help!
