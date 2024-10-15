# YOLO-augmenter
Image Augmentation for Object Detection in YOLO Format

This repository contains code and resources for augmenting various types of signs and objects onto background images, formatted for YOLO object detection training. The goal is to generate synthetic training data by overlaying objects (e.g., signs, logos, or any other custom images) onto different backgrounds. This process is useful for training deep learning models for object detection tasks in various domains such as autonomous driving, surveillance, or retail analytics.

## Features
- **Overlay Custom Objects**: Augments various custom images (e.g., signs, logos) onto different background images with automatic calculation and annotation of bounding boxes in YOLO format.
- **Perspective Transformation**: Applies random perspective transformations to objects for realistic augmentation by warping the image using a four-point transformation method.
- **Noise and Blur Effects**: Adds visual noise such as salt-and-pepper noise and Gaussian blur to simulate different environmental conditions and camera effects.
- **Grayscale and Color Variations**: Converts objects to grayscale or other color variations for diverse testing and visual variations.
- **Random Rotation**: Rotates objects at random angles to introduce diversity in orientation, enhancing the robustness of training data.
- **Resizing and Scaling**: Dynamically resizes objects proportionally before overlaying them on backgrounds, ensuring appropriate scaling based on the background size.
- **Random Occlusion**: Adds occlusions over parts of the object to simulate real-world scenarios where objects may be partially obstructed.
