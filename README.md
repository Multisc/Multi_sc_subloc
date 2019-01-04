
# Multi_sc_subloc
Prediction of protein subcellular localization based on multilayer sparse coding

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

Thank you for your advice!

