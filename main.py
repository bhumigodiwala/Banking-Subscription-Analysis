############################################################################################
#                               EE660 - Course Project
#                            Banking Subscription Analysis
#                                    Type - 1 Project
#      
#                                      Done By:
#                            Vibhav Hosahalli Venkataramaiah
#                                  Bhumi Godiwala
############################################################################################

import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import copy
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading
from scipy.stats import multivariate_normal
from numpy.random import multivariate_normal as mvn
import mvlearn
from mvlearn.semi_supervised import CTClassifier
from qns3vm.qns3vm import QN_S3VM
import pickle
import random
import joblib
import warnings
warnings.filterwarnings('ignore')

############################################################################################
## Score Metrics - Accuracy, F1 score and Confusion Matrix"""

# Function to calculate the required score metrics
def score_metrics(actual_labels,predicted_labels, show = True):
  
  #Accuracy
    sys_accuracy = accuracy_score(actual_labels,predicted_labels)

  #F1 score
    sys_f1_score = f1_score(actual_labels,predicted_labels, average='micro')
    
  #Confusion Matrix
    sys_cf = confusion_matrix(actual_labels,predicted_labels)
    
    if(show):
        print("Accuracy of system is ", sys_accuracy)
        print("F1 score of system is ", sys_f1_score)
        print("Confusion Matrix of system is \n", sys_cf)
        plt.figure()
        sns.heatmap(sys_cf,annot = True)
        plt.show()
    return accuracy_score 

# Function to define trivial system
def trivial(probability_distribution, test_labels):
  
    test_labels_np = test_labels.to_numpy()

  # Creating a Bernoulli RV with the calculated probability
    test_pred = np.zeros([len(test_labels_np)])
    for i in range(len(test_labels_np)):
        n = random.uniform(0,1)
        if(n>=probability_distribution):
            test_pred[i] = 1
        else:
            test_pred[i] = 0
    
    score_metrics(test_labels_np,test_pred)
    
print('############################################################################################ ')
print('############################################################################################')
print('#                               EE660 - Course Project')
print('#                            Banking Subscription Analysis')
print('#                                    Type - 1 Project')
print('#      ')
print('#                                      Done By:')
print('#                            Vibhav Hosahalli Venkataramaiah')
print('#                                   Bhumi Godiwala')
print('############################################################################################')

# Load the test data and process it
test = pd.read_csv('data/test_data.csv')
test_data = test.iloc[:,1:-1]
test_label = test.iloc[:,-1]


#Run the baseline model
print('For the trivial Regressor:')
prob_dist = np.load('utils/prob_distribution.npy')
trivial(prob_dist,test_label)


# Run the test scores for supervised learning methods
print('For Supervised Learning Methods:')

print('For Logistic Regression:')
log_reg_sl = pickle.load(open('utils/best_logistic_regression.sav','rb'))
log_reg_pred = log_reg_sl.predict(test_data)
score_metrics(test_label,log_reg_pred)

print('For Decision Tree Classifier:')
dt_sl = pickle.load(open('utils/best_decision_tree.sav','rb'))
dt_pred = dt_sl.predict(test_data)
score_metrics(test_label,dt_pred)

print('For the Random Forest Classifier:')
rf_sl = pickle.load(open('utils/best_random_forest.sav','rb'))
rf_pred = rf_sl.predict(test_data)
score_metrics(test_label,rf_pred)

print('For the SVM Classifier:')
svm_sl = pickle.load(open('utils/best_svm_model.sav','rb'))
svm_pred = svm_sl.predict(test_data)
score_metrics(test_label,svm_pred)

# Run the test scores fot the semi-supervised learning methods
print('For Semi-supervised Learning Methods:')

print('For Self-Training + Random Forest Classifier:')
ssl_rf = pickle.load(open('utils/self_training_random_forest.sav','rb'))
ssl_rf_pred = ssl_rf.predict(test_data)
score_metrics(test_label,ssl_rf_pred)

print('For the Co-Training Classifier:')
ctc_model = pickle.load(open('utils/co_training_model.sav','rb'))
ctc_model_pred = ctc_model.predict([test_data.to_numpy(), test_data.to_numpy()])
score_metrics(test_label.to_numpy(),ctc_model_pred)

print('For the S3VM Classifier:')
data = np.load('utils/s3vm_results.npz')
test_labels_s3vm =  data['name1']
s3vm_pred = data['name2']
score_metrics(test_labels_s3vm,s3vm_pred)

print('For the Semi-Supervised GMM model:')
data = np.load('utils/ssl_gmm_results.npz')
test_labels_gmm =  data['name1']
gmm_pred = data['nam2']
score_metrics(test_labels_s3vm,s3vm_pred)

print('Label Propogation and Label Spreading, cant run on local machines, results and code files added in report and repository\n\n')
print('############################################################################################ ')
print('The Best Performing Model: Self-Training + Random Forest')
score_metrics(test_label,ssl_rf_pred)
