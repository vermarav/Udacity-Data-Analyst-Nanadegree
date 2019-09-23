#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tester import dump_classifier_and_data
from sklearn.model_selection import GridSearchCV
from tester import test_classifier
from sklearn.metrics import accuracy_score, classification_report


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','total_stock_value', 'exercised_stock_options', 'bonus'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

### connvert NaN to 0

all_features = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'email_address', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

for key in data_dict:
    for feature in all_features:
        if data_dict[key][feature] == 'NaN':
            data_dict[key][feature] = 0

#print(data_dict)


### Task 3: Create new feature(s)
for key in data_dict:
    if data_dict[key]['from_poi_to_this_person'] !=0 and data_dict[key]['from_messages']:
        from_poi_to_persion_ratio = float(data_dict[key]['from_poi_to_this_person']/data_dict[key]['from_messages'])
        data_dict[key]['from_poi_to_persion_ratio'] = from_poi_to_persion_ratio
    if data_dict[key]['from_this_person_to_poi'] !=0 and data_dict[key]['to_messages']:
        from_this_person_to_poi_ratio  = float(data_dict[key]['from_this_person_to_poi']/data_dict[key]['to_messages'])
        data_dict[key]['from_this_person_to_poi_ratio'] = from_this_person_to_poi_ratio
    else:
        data_dict[key]['from_poi_to_persion_ratio'] = 0
        data_dict[key]['from_this_person_to_poi_ratio'] = 0





### Store to my_dataset for easy export below.
my_dataset = data_dict

#print(my_dataset.keys())

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm  import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

decsTree = DecisionTreeClassifier()
knn = KNeighborsClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7], 'max_features': ['auto', 'sqrt', 'log2', None], 'criterion': ['gini', 'entropy']}
clf = GridSearchCV(decsTree, parameters)

parameters = {'n_neighbors': [2, 3, 4, 5, 7, 9]}
clf_knn = GridSearchCV(knn, parameters)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

cv = StratifiedShuffleSplit(n_splits=1000, random_state=42, test_size=0.3) 

features_train = []
features_test  = []
labels_train   = []
labels_test    = []

for train_idx, test_idx in cv.split(features, labels):
    #for train_idx, test_idx in cv: 
        
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

print('----------------------------------------------------------------')
print(len(features_train), len(labels_train))
print(len(features_test), len(labels_test))
print('----------------------------------------------------------------')

#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
clf = clf.best_estimator_

clf_knn.fit(features_train, labels_train)
clf_knn = clf_knn.best_estimator_


estimators_knn = [('scaler', MinMaxScaler()), ('reduce_dim', PCA()), ('clf', clf_knn)]

estimators = [('reduce_dim', PCA()), ('clf', clf)]

clf = Pipeline(estimators)

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)



clf_knn = Pipeline(estimators_knn)
clf_knn.fit(features_train, labels_train)
pred_knn = clf_knn.predict(features_test)





print(accuracy_score(labels_test, pred))
#print(classification_report(labels_test, pred))

test_classifier(clf,my_dataset,features_list)

#print('----------------------------------------------------------------')

#print(classification_report(labels_test,pred_svc))
#test_classifier(clf_svc,my_dataset,features_list)


print('----------------------------------------------------------------')

print(accuracy_score(labels_test, pred_knn))
#print(classification_report(labels_test,pred_knn))
test_classifier(clf_knn,my_dataset,features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)