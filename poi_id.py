#!/usr/bin/python

import sys
import pickle
#sys.path.append("../tools/")
import pandas as pd
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif# Perform feature selection
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
                'director_fees', 'to_messages', 'from_poi_to_this_person', 
                'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 
                'fraction_from_poi_to_this_person','fraction_from_this_person_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

enron_data = pd.DataFrame.from_dict(data_dict, orient = 'index')
enron_data = enron_data.replace('NaN', np.nan)
#replace NaN to 0
enron_data = enron_data.fillna(0)

#Task 2: Remove outliers
enron_data = enron_data.drop("LOCKHART EUGENE E")
enron_data = enron_data.drop("THE TRAVEL AGENCY IN THE PARK")
enron_data = enron_data.drop("TOTAL")
enron_data = enron_data.drop(columns=['email_address'])

### Task 3: Create new feature(s)
#1 new feature 'ratio_from_poi_to_this_person'
enron_data['fraction_from_poi_to_this_person'] = enron_data["from_poi_to_this_person"].\
divide(enron_data["to_messages"], fill_value = 0).fillna(0)

enron_data['fraction_from_this_person_to_poi'] = enron_data["from_this_person_to_poi"].\
divide(enron_data["from_messages"], fill_value = 0).fillna(0)


### Store to my_dataset for easy export below.
#data_dict = enron_data.to_dict('index')
my_dataset = enron_data.to_dict('index')

### Feature selection
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

selector = SelectKBest(f_classif)
_ = selector.fit(features, labels)
# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)
feature_score = pd.DataFrame({'feature':features_list[1:], 'score':scores})
features_list = ['poi'] + feature_score.loc[feature_score['score'] > 2, 'feature'].tolist()


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

pca = PCA(n_components=3)
_ = pca.fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)

clf.fit(features_train, labels_train)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)