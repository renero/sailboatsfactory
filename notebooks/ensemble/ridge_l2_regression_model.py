
# coding: utf-8

# # Predicting the CLOSE value from the LSTM predictions
# 
# This notebook will reproduce the steps for a REGRESSION on  predictions.
# The main objective is to predict the variable actual.
# 
# Model Ridge (L2) regression, trained on 2018-11-04 10:47:55.

# Let's start with importing the required libs, and tune pandas display options:

# In[1]:


import numpy as np
import pandas as pd
import sklearn as sk
from collections import defaultdict, Counter


# In[2]:


pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# #### Importing base data
# The first step is to get our machine learning dataset:

# In[3]:


ml_dataset = pd.read_csv('/Users/renero/Documents/SideProjects/SailBoatsFactory/data/predictions.csv')
print('Base data has %i rows and %i columns' % (ml_dataset.shape[0], ml_dataset.shape[1]))
# Five first records",
ml_dataset.head(5)


# #### Initial data management
# The preprocessing aims at making the dataset compatible with modeling. At the end of this step, we will have a matrix of float numbers, with no missing values. We'll use the features and the preprocessing steps defined in Models.
# 
# Let's only keep selected features

# In[4]:


ml_dataset = ml_dataset[[u'actual', u'10yw7', u'1yw7', u'1yw3', u'1yw10', u'median', u'5yw10', u'10yw3', u'5yw3', u'avg', u'5yw7']]
# Five first records",
ml_dataset.head(5)


# Let's first coerce categorical columns into unicode, numerical features into floats.

# In[5]:


# astype('unicode') does not work as expected
def coerce_to_unicode(x):
    if isinstance(x, str):
        return unicode(x,'utf-8')
    else:
        return unicode(x)

categorical_features = []
numerical_features = [u'10yw7', u'1yw7', u'1yw3', u'1yw10', u'median', u'5yw10', u'10yw3', u'5yw3', u'avg', u'5yw7']
text_features = []

for feature in categorical_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in text_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in numerical_features:
    if ml_dataset[feature].dtype != np.dtype('M8[ns]'):
        ml_dataset[feature] = ml_dataset[feature].astype('double')


# We renamed the target variable to a column named target

# In[6]:


ml_dataset['__target__'] = ml_dataset['actual']
del ml_dataset['actual']

# Remove rows for which the target is unknown.
ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]


# #### Cross-validation strategy
# The dataset needs to be split into 2 new sets, one that will be used for training the model (train set) and another that will be used to test its generalization capability (test set).
# 
# This is a simple cross-validation strategy.

from sklearn.model_selection import train_test_split

train, test = train_test_split(ml_dataset, test_size=0.2, shuffle=False)
print('Train data has %i rows and %i columns' % (train.shape[0], train.shape[1]))
print('Test data has %i rows and %i columns' % (test.shape[0], test.shape[1]))


# #### Modeling

# Before actually creating our model, we need to split the datasets into their features and labels parts:

train_X = train.drop('__target__', axis=1)
test_X = test.drop('__target__', axis=1)

train_Y = np.array(train['__target__'])
test_Y = np.array(test['__target__'])


# Now we can finally create our model !

from sklearn.linear_model import RidgeCV
clf = RidgeCV(fit_intercept=True, normalize=True)

# ... And train it

clf.fit(train_X, train_Y)


# Build up our result dataset

_predictions = clf.predict(test_X)
predictions = pd.Series(data=_predictions, index=test_X.index, name='predicted_value')

# Build scored dataset
results_test = test_X.join(predictions, how='left')
results_test = results_test.join(test['__target__'], how='left')
results_test = results_test.rename(columns= {'__target__': 'actual'})


# #### Results
# You can measure the model's accuracy:

c =  results_test[['predicted_value', 'actual']].corr()
print('Pearson correlation: %s' % c['predicted_value'][1])

# I measure the score of the model over the test sets, as indicated in the Ridge SKLearn manual

score = clf.score(test_X, test_Y)
print("Test score: {0:.2f} %".format(100 * score))


# I dump the model to a pickle file, so that I can use it from the main code.

# In[ ]:


import pickle
pkl_filename = "/Users/renero/Documents/SideProjects/SailBoatsFactory/networks/ridge_l2_model.pkl"  
with open(pkl_filename, 'wb') as file: 
    pickle.dump(clf, file)


# I check that model still works

# In[ ]:


pkl_filename = "/Users/renero/Documents/SideProjects/SailBoatsFactory/networks/ridge_l2_model.pkl"  
with open(pkl_filename, 'rb') as file:  
    pickle_model = pickle.load(file)
new_score = pickle_model.score(test_X, test_Y)
print("Test score: {0:.2f} %".format(100 * new_score))

