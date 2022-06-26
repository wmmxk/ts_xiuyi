# https://www.baeldung.com/cs/svm-multiclass-classification

# check the classification performance using all the features
# check whether it is the same as the output you used TS_classify in matlab
# check classification performance if you used fewer features, such as the half
# check classification performance if you used the few PCA features

# the input is feature matching the task is  sub-10, the feature classification task
# accuracy is 40.3146;

import os
import pandas as pd
import numpy as np
import mat73
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA

def svm_classify(X,y):
    """
    define a function to run multiclass classification
    :param X: features set
    :param y: class label
    :return: poly_accuracy, poly_f1
    """

    # divide the dataset to 80% for training, 20% for resting
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20,
                                                                        random_state=101)
    # create a svm with Polynomial kernel
    # rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)  # does not work well
    poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)

    # test the classifier using the test data set
    poly_pred = poly.predict(X_test)

    # calculate the accuracy and f1 scores
    poly_accuracy = accuracy_score(y_test, poly_pred)
    poly_f1 = f1_score(y_test, poly_pred, average='weighted')
    print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy * 100))
    print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1 * 100))
    print('       ')

    return poly_accuracy, poly_f1

#%% step 1: prepare the feature vectors and labels
# get the path of the current code

path_curr = os.getcwd()

file_data = os.path.join(path_curr,'data_features.mat')
file_label = os.path.join(path_curr,'labels.xlsx')

# read the label file and get the label of each sample
data_label = pd.read_excel(file_label)
labels_each = data_label['network_id']
labels = list(labels_each)*4

y = labels

# load data of feature vectors
data_all = mat73.loadmat(file_data )
data = data_all['TS_DataMat']


#%% step 2: run the classification using all the features
data_all_features = data
print ('classify using all the features')
accuracy_all_features, f1_all_features = svm_classify(data_all_features,labels)

#%% step 3: run the classification using the part of the features

# identify distinct values by feature
df = pd.DataFrame(data, columns = list(np.arange(1,data.shape[1]+1)))
distinct_counter = df.apply(lambda x: len(x.unique()))

# sort the distinct_counter based on the unique number
distinct_counter.sort_values(ascending = False, inplace=True)

cols_index = list(distinct_counter.index)
# choose the first * features for classification
for feature_part_num in range(500,7000,500):
    data_part = df[cols_index[0:feature_part_num]]

    # classify
    print ('classify using %s features'%(feature_part_num))
    accuracy_part_features, f1_part_features = svm_classify(data_part.values, labels)

#%% run PCA to choose the top features
n_components = 1600
whiten = False
random_state = 2018
pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
data_PCA = pca.fit_transform(data)

# percentage of variance captured by %s n_components
print ('variance explained by the %s principal components %s' %(n_components, sum(pca.explained_variance_ratio_)))

# percentage of variance captured by x principal components

for pc_num in range(200,1600,200):
    print ('variance explained by the first %s PCs: %s'%(pc_num, np.sum(pca.explained_variance_ratio_[0:pc_num])))

# classification using the PCA features
for PCs_num in range(500,1600,200):
    print ('classify using %s principal components'%(PCs_num))
    data_PCA_part = data_PCA[:,0:PCs_num]
    accuracy_part_PCs, f1_part_PCs = svm_classify(data_PCA_part, labels)
    
print ('well done')