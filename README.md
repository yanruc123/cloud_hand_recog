# Cloud Hand Recognition Project

This project is a full-stack, machine learning-powered game deployed on Google Cloud with Kubernetes, using Locust for traffic testing. The game utilizes both non-machine learning and machine learning methods for hand gesture recognition, implemented with OpenCV and TensorFlow.

## Overview

The project involves the use of Python, TensorFlow, AutoML API, OpenCV, Google Cloud, Kubernetes, and Locust. The machine learning model is trained to learn gestures and make predictions with almost 100% accuracy. The project includes both local and cloud deployment.

## Features

1. **Hand Gesture Recognition**: Implemented and compared non-ML computer vision methods and ML methods for hand gesture recognition, resulting in a 5% improvement in prediction accuracy.

2. **Machine Learning Model**: The local model uses Keras and TensorFlow with 4 convolutional layers, while the cloud model uses AutoML API.

3. **Deployment**: The game is deployed both locally and on the cloud. The local deployment uses the OpenCV library for image processing, while the cloud deployment uses Google Cloud as the deployment platform and locust/kubernetes for load testing.

## Running the Project

For local deployment, run `game.py` on your local machine with Python 3.7 and the required libraries. For cloud deployment, use [this link](https://cloud-hand-recog-285002.uc.r.appspot.com/).

## Recognition

The project's abstract was accepted by the 2020 3rd International Conference on Virtual Reality Technology.

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.
