# facenet_sarthak
Facial attributes Classification using CNNs

- Project: Multiple Facial Attributes Prediction using Deep Learning techniques
- Implemented a deep convolutional neural network (CNNs) to predict facial attributes. Analyzed impact of bias of 73 features on specific layers. Used transfer learning on the facenet embeddings to improve the accuracy of the results 

Run Sarthak_lateest/Latest :

python tensorflow_train.py /DATA-2/Datasets_2/lfw_test2 /DATA-2/Datasets_2/lfw/Alter/20180402-114759/20180402-114759.pb --batch_size 12 --image_size 160 --min_nrof_images_per_class 1 --nrof_train_images_per_class 530
