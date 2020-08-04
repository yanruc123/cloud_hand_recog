# cloud_hand_recog

--used machine learning (TensorFlow, AutoML API) to learn gestures and make predictions with almost 100% accuracy 

--used Google Cloud to deploy the website

--used kubernetes and locust for load testing


see deployment: https://cloud-hand-recog-285002.uc.r.appspot.com/

This repo includes both local deployment and cloud deployment:

Local deployment: use OpenCV library for image processing

Cloud deployment: use Google Cloud for deployment platform and locust/kubernetes for load testing

Machine Learning Model: 
  
  local: Keras, Tensorflow -- 4 convolutional layers
  
  cloud: AutoML API
