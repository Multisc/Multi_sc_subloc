
# Multi_sc_subloc
Prediction of protein subcellular localization based on multilayer sparse coding


The purpose of this open source is that we want to offer an effective feature extraction method for you to complete the recognition of different protein sequences and then to find the relationship between them. So if you have you own data sets and the data size is enough big, we prefer to encourage you to train you own model using our algorithms, and we think that will obtain a good result. If you just want to predict the subcellular locations of a few protein sequences, we recommend you to use the model trained by CH317(the bigger datasets used in our experiment), because statistically speaking, the more data you have, the more accurate the model is. And we have tested the performance of 2 datasets on each other, actually there are only subtle differences, less than 1%, the model of CH317 is better than it of ZD98, and the specific web-server is still in development and will be uploaded on github as soon as possible. 

Based on the traditional protein sequence feature extraction algorithm AAC, we introduced sparse coding to optimize sequence features, and proposed a feature fusion method based on multi-level dictionary. The main contribution includes: firstly using sliding window segmentation to extract the sequence fragments of protein sequences, and the traditional feature extraction algorithm was used to encode them. Then the K-SVD algorithm was used to learn the dictionary, and the sequence feature matrix was sparsely represented by the OMP algorithm. The feature representation based on different sizes of dictionaries are mean-pooled to help extract the overall and local feature information. Finally the SVM multi-class classifier is used to predict the subcellular location of the proteins.

STEP 1:
Change your data format into csv file, one file represents the protein sequences, and the other represents the corresponding labels.

STEP 2:
Using sliding window segmentation to extract the sequence fragments of protein sequences, the code file is cut_piece.py and piece_aac.py.
There are a parameter that you need to set by yourself according to your experimental result, that is, the window size.

STEP 3:
Using spare coding.py. You just need to set the size of the dictionary, we have write the max_pooling into the file, so by this step you can obtain your final feature vectors directly.

Step 4:
Classifier. You can use your own classifier, but we use libSVM as our final classifier, you can see it in Classifier file.

Compared with other experimental results with the same support vector machine classifier, the experimental results show that the proposed method can not only simplify the feature extraction process, reduce the time and space complexity of the classifier, but also reflect the sequence features more comprehensively and improve the classification performance. 


