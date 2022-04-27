# Real-time Potholes Detection: An AIoT application facilitating road safety and city micro-mobility schemes


### I. Why Detecting Potholes?
Road infrastructure is playing an imperative role in achieving the United Nations’ Sustainable Development Goal of providing access to safe, affordable, accessible and sustainable transport systems for all, improving road safety, notably (United Nations, 2015).
At the city scale, metropolises like London are facilitating micro-mobility diversity including launching bike and e-scooter sharing schemes that will also benefit from the high-quality road infrastructure (London City Hall, 2015; Transport for London).
However, London’s councils spent a total of £17.9 million on fixing potholes in 2020 according to an information request to the government (Anahita Hossein-Pour, 2021). It is also recorded that London is only the 62nd best city worldwide for cycling based on data related to infrastructure, road quality, accidents and bike-sharing schemes (luko, 2022). It is generally believed that examination and maintenance of road quality cost an amount of time and money from the local authorities.
In conclusion, in order to provide an instant and sustainable service of road repairs, collaborative crowdsourcing via the mobile app seems to be one of the solutions to continuously share the latest information on potholes, helping improve the efficiency of operation.

<img width="700" alt="image" src="https://user-images.githubusercontent.com/52306317/165566024-991fea3f-16b8-41a9-976b-54e4c001d0f8.png">


### II. Research Question
Can potholes be detected based on real-time images input by using a deep learning model on an edge device?

### III. Application Overview
The purpose of this application is to classify if it is a pothole or a plain road. The image input relies on the camera on the edge device–– in this case would be a mobile phone. While the image is captured, the TensorFlow model will run the image classification model which is a deep learning model built, trained and converted to the TensorFlow Lite model in the Colab environment. The model will use real-time image input as an inference to predict the possibilities of two categories of potholes and plain roads respectively and to present the results on the mobile app automatically.


<img width="700" alt="image" src="https://user-images.githubusercontent.com/52306317/165565335-d07adbf6-01b8-44ea-86b7-7abb632ebe05.png">

Figure 2. Image classification model

### IV. Data
Initially, two Kaggle datasets about potholes and plain roads images are collected (Kaggle, 2020a, 2020b). However, since the original data of plain roads have similar shooting angles and compositions and contain too much unnecessary information like the sky and grass, additional image processing is conducted to reposition and resize images so that they better present the traits of normal roads. This step is crucial as the project would like to deploy a deep learning model in the real world. By doing so the usability of the application is high and reliable even with different shooting angles and environmental settings. All the processing workflows were done by using the Pillow package and Python scripts. In addition, more images are added to the dataset by myself taking pictures in the streets and from Google Image.
Regarding the data sizes of these two categories, they were unbalanced in the beginning–– The size of pothole data is 3 times more than the one for the plain road which will influence the training process and result in a biased accuracy rate. Therefore, pothole data is randomly chosen to reduce the size of the dataset while the project might want to increase the size of the plain road dataset instead in the future.
The final dataset contains 400 images of potholes and 400 images of plain roads. The data was further split into 60% (480 images) for the training dataset, 20% (160 images) for validation and 20% (160 images) for testing.

<img width="700" alt="image" src="https://user-images.githubusercontent.com/52306317/165565768-7db304d0-2981-4de2-80aa-d3652218ccac.png">

Figure 3. Original data collected from two Kaggle datasets

<img width="700" alt="image" src="https://user-images.githubusercontent.com/52306317/165566082-559dbbae-463c-429e-ac8d-bd8ae01ca484.png">


Figure 4. Image processing by using the Pillow package

<img width="700" alt="image" src="https://user-images.githubusercontent.com/52306317/165566700-14d8a977-a6ce-4d98-896e-812af311b15b.png">


Figure 5. Final dataset and the split for training, validation, and test subsets

### V. Model
In order to obtain the best accuracy score on the testing set, three different ways of modelling were taken done with a similar test environment. All of them are based on the Keras Framework; the first one is Keras sequential model, the second one is Transfer Learning, and the last one is Fine-tuning the pre-trained model. The 11 trials detailed in the below table show that Inception V3 has the best performance on the testing data.
For the transfer learning, the pre-trained model and its weights will be loaded and frozen, and then a flatten layer and a dense layer will be added on top of that base model to get the result of potholes detection. More tested parameters for the Inception V3 model will be elaborated on in the next section. As for the fine-tuning model based on Inception V3, it is found that the experiments this project has done do not improve the accuracy scores on both validation and testing data. One of the possible reasons is that the new dataset is smaller and different from the pre-trained dataset. Inception V3 was pre-trained using a dataset of 1,000 classes from the original ImageNet dataset which has over 1 million training images (Intel, 2019). However, the only related class to this project is the images of the manhole covers (Emily Fox and Carlos Guestrin, 2015). According to the research, in this case, it might be hard to find the best number of layers to be re-trained so more experiments would be required, or more new data would be added to increase the size (Marcelino, 2018). Given the limited time of the project, transfer learning which only turns the generic old features into predictions on a new dataset is good enough to go. More performance metrics can be found in the next section.

Table 1. Settings for the test environment

<img width="700" alt="Screenshot 2022-04-27 at 17 28 36" src="https://user-images.githubusercontent.com/52306317/165567339-d8179fc3-7a9f-400d-bdb7-36c080ef0b3a.png">



Table 2. Model testing based on the similar test environment

<img width="700" alt="Screenshot 2022-04-27 at 17 31 07" src="https://user-images.githubusercontent.com/52306317/165568120-af02e928-85f6-4cb3-9812-849c65fbda66.png">

<img width="700" alt="image" src="https://user-images.githubusercontent.com/52306317/165568958-380e52b0-8288-4741-af96-b410db097f95.png">

Figure 6. Model information on the transfer learning 

<img width="700" alt="image" src="https://user-images.githubusercontent.com/52306317/165569030-53b9f74e-7950-4357-852f-4383497e2281.png">

Figure 7. Schematic Diagram of Inception V3
(Google Cloud, 2022)


### VI. Experiments

There are 13 parameters shown in Table 3 that are tested in the Colab environment and Python script to obtain the best accuracy score on the testing dataset. Since the Inception V3 model has been chosen in the prior step, the following experiment focuses on optimising the number of epochs, batch size, and image colour channel. In addition, it is proven that using the data augmentation function will increase the accuracy of the testing data by 3.4%. Another testing result shows that Grayscale images lead to a 50% accuracy score where most potholes were detected as plain roads. In the end, Inception V3 with 64 epochs, 64 examples per iteration (batch size) and RGB images are applied to the model, making a 97.7% accuracy on training data, 96.9% on validation data, 95.0% on testing data. (* Fine-tune Optimizer = SGD(lr=0.0001, momentum=0.9)
A confusion matrix is applied to further understand the performance of the model. In the case of this application, Recall is worth investigating since the application wants to detect as many potholes correctly as possible. Recall means the ratio of predicted positive to the total real positive cases while Precision means the ratio of real positive cases to the total positive predictions. Fortunately, the Recall of potholes achieves 99%, presenting a high capability of detecting potholes on the testing data. However, the risk of misprediction lies in the 92% of Precision score on potholes, resulting in examiners will take more time and effort to check the situation and report errors.
The last work in the Colab is to convert the TensorFlow model into the TensorFlow Lite float model and then into the TensorFlow Lite quantization model which reduces the memory requirement and computational cost of using a neural network.

Table 3. Experimental parameters and results


<img width="700" alt="Screenshot 2022-04-27 at 17 35 12" src="https://user-images.githubusercontent.com/52306317/165570272-781370de-a57b-43e3-aa99-cae872e3eb6f.png">



Table 4. Fine-tuning of InceptionV3 model

<img width="700" alt="Screenshot 2022-04-27 at 17 35 50" src="https://user-images.githubusercontent.com/52306317/165570720-4142d3f9-5f0b-4288-b806-dd81ccef0356.png">


<img width="700" alt="image" src="https://user-images.githubusercontent.com/52306317/165570945-ae1bc2d8-0ca2-41c4-9680-93f2992f0ff7.png">

Figure 8. Training and validation accuracy and loss of final model


Table 5. Accuracies of each dataset

<img width="700" alt="Screenshot 2022-04-27 at 17 37 46" src="https://user-images.githubusercontent.com/52306317/165571745-c09284c3-5738-49e0-8b79-fb7941e35cb9.png">



Table 6. Confusion matrix for the testing dataset

<img width="700" alt="Screenshot 2022-04-27 at 17 37 55" src="https://user-images.githubusercontent.com/52306317/165571846-471399ec-f7d9-4543-b874-8c2c2e5b3449.png">


Table 7. Metrics of a confusion matrix for the testing dataset

<img width="700" alt="Screenshot 2022-04-27 at 17 38 14" src="https://user-images.githubusercontent.com/52306317/165572059-21761dbf-21b6-4073-92e3-bc722513bdda.png">


Table 8. Model sizes of TF model, TF Lite float model, and TF Lite quantization model

<img width="700" alt="Screenshot 2022-04-27 at 17 38 24" src="https://user-images.githubusercontent.com/52306317/165572171-421472d0-d73c-4b08-b8ca-a04606bab259.png">



### VII. Real-world Application
There are two main reasons for deploying an image classification model to mobile phones instead of other microcontrollers. Firstly, a further integrated AIoT mobile app can be developed to help automatically record, notify, and report potholes’ locations and their images to local authorities and other road users. Cyclists, e-scooter riders, motorcyclists, or other micro-mobility users can easily attach their phones to vehicles, taking pictures of potholes and reporting them to the platform. Also, the average speed of urban micro-mobility is able to maintain the quality of inference.
Secondly, considering the complexity of an image classification model, mobile phones seem to be more flexible in model size, rapid in inference, and capable of high-resolution photo shooting and Bluetooth communication.
The real-world experimental results show that the app works well during the daytime and the predicted probability of plain road might be slightly influenced by shadows yet the prediction result is still correct. Current issues include that part of drain covers are be classified as potholes and sometimes repaired potholes still can be seen as potholes if they don’t look smooth, but they actually no longer have any safety issues.
To see the demo video, please visit https://www.youtube.com/watch?v=xIY2LgA5Sgo (start from 02:21)

<img width="700" alt="image" src="https://user-images.githubusercontent.com/52306317/165572374-4d0b9d2d-ac18-4b58-a3e6-57ad9572237a.png">

Figure 9. Workflow of the TF Lite model deployment 


Figure 10. Application scenarios

Figure 11. The real-world application- shooting potholes, plain road and drain covers

Figure 12. The real-world application- shooting drain covers and repaired pothole


### VIII. Results and Observations
1.	Data Collection
More data collected from different environmental settings, such as pictures taken at night, on rainy days, and for different pavements like stone roads, may increase the usability of the application.
2.	Image Processing
Since most potholes have evident edges, edge sharpening or contouring may help detect the sizes and shapes of potholes providing more specific information to record and use.
3.	Model and Experiment
First of all, although the image classification model this project has built has a good performance in a real-world application, the object detection model could be considered in the future because other objects on the road such as drain covers may influence the inferencing accuracy.
The pre-trained models (VGG16, ResNet50, Inception V3 and MobileNet V2) the project has tested are all trained on the ImageNet dataset. One of the possible reasons for the higher accuracy score on the Inception V3 model is that although the salient parts (potholes) vary in size and locations between the images, the Inception V3 model adopts multiple sizes of kernels operating on the same level, solving the problem of choosing the right kernel size; meanwhile, making the network get a bit “wider” rather than “deeper” to save more computational cost (Raj, 2020).
In addition, after processing images the accuracy of real-world application significantly increases because the new model eliminates the limitations on specific camera angles and compositions.
4.	Real-world Application
After the initial trials, several observations were found as below:
•	The height of the camera should be 50~200 cm
•	The maximum distance of shooting is 7 m
•	The filming angles should be depression of 20º~90º
•	The average latency which is the amount of time it takes to run a single inference with a given model is 19 ms; however, the speed of moving vehicles should be slower than 5 m/s which is also below the definition of micro-mobility being less than 25 km/h (= 15 mph = 6.7 m/s).

### IX. Bibliography
Anahita Hossein-Pour (2021) The London boroughs with the most and least potholes ranked, MyLondon. Available at: https://www.mylondon.news/news/west-london-news/london-boroughs-most-least-potholes-20293869.
Emily Fox and Carlos Guestrin (2015) ImageNet 1000 class id to human readable labels, Gist. Available at: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a.
Google Cloud (2022) Advanced Guide to Inception v3 | Cloud TPU, Google Cloud. Available at: https://cloud.google.com/tpu/docs/inception-v3-advanced.
Intel (2019) Inception V3 Deep Convolutional Architecture For Classifying Acute..., Intel. Available at: https://www.intel.com/content/www/us/en/developer/articles/technical/inception-v3-deep-convolutional-architecture-for-classifying-acute-myeloidlymphoblastic.html.
Kaggle (2020a) Annotated Potholes Image Dataset. Available at: https://www.kaggle.com/chitholian/annotated-potholes-dataset.
Kaggle (2020b) Pothole and Plain Road Images. Available at: https://www.kaggle.com/virenbr11/pothole-and-plain-rode-images.
London City Hall (2015) Policy 6.9 Cycling, Policy 6.9 Cycling. Available at: https://www.london.gov.uk//what-we-do/planning/london-plan/past-versions-and-alterations-london-plan/london-plan-2016/london-plan-chapter-six-londons-transport/poli-0.
luko (2022) Global Bicycle Cities Index 2022, luko. Available at: https://de.luko.eu/en/advice/guide/bike-index/.
Marcelino, P. (2018) Transfer learning from pre-trained models, Medium. Available at: https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751.
Raj, B. (2020) A Simple Guide to the Versions of the Inception Network, Medium. Available at: https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202.
Transport for London (no date) Electric scooters, Electric scooters. Available at: https://www.tfl.gov.uk/modes/driving/electric-scooter-rental-trial.
United Nations (2015) SDG Indicators, Sustainable Development Goals. Available at: https://unstats.un.org/sdgs/metadata/?Text=&Goal=&Target=11.2.

<img width="436" alt="image" src="https://user-images.githubusercontent.com/52306317/165565279-1859e0b1-4c7f-46b1-b73b-8b05faf00d80.png">
