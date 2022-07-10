## **RepMet**

### **Content**

This repository contain an implementation of few-shot learning neural network called *RepMet* [1] used for an classification task of particular instances.

### **Model and Training**

RepMet (*Representative-based metric learning*) is a sub-net architecture for jointly training an embeding space and a set of mixture distributions. This approach falls within the field of Distance Metric Learning (DML) and aims to learn a series of representatives that contain information about an object in an embedding space, which are useful when classifying the object with few-shot learning. Next you may visualize the arhictecture of RepMet:

<img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/images/repmet.png" height="400" width ="400" />

The neural network architecture correspond an a inception V3 as a Backbone and a MLP network of three layers to code a embedded feature vector.

The training regime consider a *N-way K-shot* training procedure, where N is the number of classes of each training episode and K means the number of instances of each class for each training episode.

### **Dataset**

The datasets used in this project were: *Stanford-Dogs*[2] and *LaSOT* [3]. Both used for few-shot learning or fine-grained image categorization. The first contains 120 breeds of dogs and it's possible to download it from [here](http://vision.stanford.edu/aditya86/ImageNetDogs/). An example of this dataset is shown below:

<img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/images/StanfordDogs.png" height="400" width ="400" />

The second dataset contains several videos of different objects, which are categorized in different classes. Specifically, contains object like bottle, cup, rubicCube, some animals and many more. You can get this data from [here](http://vision.cs.stonybrook.edu/~lasot/). An example of this dataset is shown below:

<img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/images/LaSOT.png" height="400" width ="400" />

### **Results**

As a first result, the accuracy and the loss function of the classification task of the StanfordDogs dataset are shown:

<img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/images/resultados_dogs.png" height="400" width ="400" /> <img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/images/resultados_dogs_loss.png" height="400" width ="400" />

Second, the accuracy on the LaSOT dataset with and without rotation is shown:

<img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/images/lasot-m_acc_1.png" height="400" width ="400" /> <img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/images/lasot-m_acc_2.png" height="400" width ="400" />

And the respective confusion matrix:

<img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/images/cm_lasot-m_1.png" height="400" width ="400" /> <img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/images/cm_lasot-m_2.png" height="400" width ="400" />


### Reference

[1] Karlinsky, L.; Shtok, J.; Harary, S.; Schwartz, E.; Aides, A.; Feris, R.; Giryes, R.; and Bronstein, A. M. 2019. RepMet: Representative-based metric learning for classification and one-shot object
detection. CVPR 5197–5206. URL: http://arxiv.org/abs/1806.04728.

[2] Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset
for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization
(FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.

[3] H. Fan et al., LaSOT: A High-Quality Benchmark for Large-Scale Single Object Tracking,2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA,
USA, 2019, pp. 5369-5378, doi: 10.1109/CVPR.2019.00552.