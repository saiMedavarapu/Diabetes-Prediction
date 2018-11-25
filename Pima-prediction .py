#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[2]:


import pandas as pd #data frame library
import matplotlib.pyplot as plt        #plots data
import numpy as np                     # Math support
import os


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# ## Load and review data



df = pd.read_csv(r'/Users/Vaibhav/Desktop/ML/MachineLearningWithPython-master/Notebooks/data/pima-data.csv')


# In[21]:


df.shape


# In[22]:


df.head(5)


# In[23]:


df.tail(5)


# In[24]:


df.isnull().values.any()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Check

# In[68]:


df.shape


# In[ ]:





# In[13]:


df = pd.read_csv(r'/Users/Vaibhav/Desktop/ML/MachineLearningWithPython-master/Notebooks/data/pima-data.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


df.shape


# In[19]:


df.head(5)


# In[20]:


df.tail(6)


# In[21]:


df.isnull().values.any()


# In[5]:


df.head(15)


# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


corr = df.corr()


# In[ ]:





# In[ ]:





# In[29]:


import pandas as pd #data frame library
import matplotlib.pyplot as plt        #plots data
import numpy as np                     # Math support
import os


get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


df.tail(6)


# In[ ]:





# In[33]:


df.corr()


# In[34]:


df.head()


# In[35]:


del df['skin']


# In[36]:


df.head()


# In[ ]:





# In[38]:


plt.show()


# In[ ]:





# In[40]:


import matplotlib
import matplotlib.pyplot as plt


# In[ ]:





# In[42]:


plt.show()


# In[43]:


plt.figure()


# In[44]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[4]:


df = pd.read_csv(r'/Users/Vaibhav/Desktop/ML/MachineLearningWithPython-master/Notebooks/data/pima-data.csv')


# In[5]:


df.head()


# In[6]:


def plot_corr(df, size =11):
    corr= df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)),corr.columns)
    plt.yticks(range(len(corr.columns)),corr.columns)


# In[7]:


plot_corr(df)


# In[8]:


df.corr()


# In[9]:


df.head()


# In[10]:


del df['skin']


# In[11]:


df.head()


# In[12]:


plot_corr(df)


# In[13]:


df.head()


# ##  Check DataTypes

# In[14]:


df.head()


# ## Change 0 to false and 1 to true

# ## use diabetes map method to change the values

# In[15]:


diabetes_map = {True: 1, False : 0}


# In[16]:


df['diabetes'] = df['diabetes'].map[diabetes_map]


# In[17]:


df['diabetes']=df['diabetes'].map[diabetes_map]


# In[18]:


df['diabetes']=df['diabetes'].map(diabetes_map)


# In[19]:


df.head()


# # Check True false ratio

# In[20]:


num_true = len(df.loc[df['diabetes']]== True)
num_false = len(df.loc[df['diabetes']]== False)
print("Number of true cases : {0} ({1:2.2f}%)".format(num_true/(num_true+num_false)*100))
print("Number of false cases : {0} ({1:2.2f}%)".format(num_false/(num_true+num_false)*100))


# In[21]:


num_true = len(df.loc[df['diabetes']]== True)
num_false = len(df.loc[df['diabetes']]== False)
print("Number of true cases : {0} ({1:2.2f}%)".format(num_true, (num_true/(num_true+num_false))*100))
print("Number of false cases : {0} ({1:2.2f}%)".format(num_false, (num_false/(num_true+num_false))*100))


# In[27]:


num_true = len(df.loc[df['diabetes']]== True)
num_false = len(df.loc[df['diabetes']]== False)
print("Number of true cases : {0} ({1:2.2f}%)".format(num_true, (num_true/(num_true+num_false))*100))
print("Number of false cases : {0} ({1:2.2f}%)".format(num_false, (num_false/(num_true+num_false))*100))


# In[23]:


num_true = len(df.loc[df['diabetes']]== True)
num_false = len(df.loc[df['diabetes']]== False)
print num_true


# num_true = len(df.loc[df['diabetes']]== True)
# num_false = len(df.loc[df['diabetes']]== False)
# print (num_true)

# In[25]:


num_true = len(df.loc[df['diabetes']]== True)
num_false = len(df.loc[df['diabetes']]== False)
print (num_true)
print(num_false)


# ## Splitting the data set
# ### 70% for training, 30% for testing

# In[28]:


from sklearn.cross_validation import train_test_split
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin','bmi', 'diab_pred', 'age']
predicted_class_names= ['diabetes']

x=df[feature_col_names].values #predictor feature columns
y= df[predicted_class_names].values #predicted class for true or false
split_test_size = 0.30 #splits into 30% of the testing size

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = split_test_size, random_state=42)


# In[31]:


from sklearn.model_selection import train_test_split

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin','bmi', 'diab_pred', 'age']
predicted_class_names= ['diabetes']

x=df[feature_col_names].values #predictor feature columns
y= df[predicted_class_names].values #predicted class for true or false
split_test_size = 0.30 #splits into 30% of the testing size

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = split_test_size, random_state=42)


# In[32]:


print("{0:0.2f}% in training set" .format((len(x_train)/len(df.index))*100))
print("{0:0.2f}% in test set" .format((len(x_test)/len(df.index))*100))


# #### Hidden missing values

# In[33]:


df.head()


# ## Skin thickness of 0 is not possible

# In[34]:


print("# rows in the dataframe {0} ".format(len(df)))
print("# rows missing in glucose_conc {0} ".format(len(df.loc[df['glucose_conc']=='0'])))
print("# rows missing in diastolic_bp {0} ".format(len(df.loc[df['diastolic_bp']=='0'])))
print("# rows missing in Thickness {0} ".format(len(df.loc[df['thickness']=='0'])))
print("# rows missing in insulin {0} ".format(len(df.loc[df['insulin']=='0'])))
print("# rows missing inbmi {0} ".format(len(df.loc[df['bmi']=='0'])))
print("# rows missing in diab_pred {0} ".format(len(df.loc[df['diab_pred']=='0'])))
print("# rows missing in age {0} ".format(len(df.loc[df['age']=='0'])))


# In[36]:


print("# rows in the dataframe {0} ".format(len(df)))
print("# rows missing in glucose_conc {0} ".format(len(df.loc[df['glucose_conc']==0])))


# In[37]:


print("# rows in the dataframe {0} ".format(len(df)))
print("# rows missing in glucose_conc {0} ".format(len(df.loc[df['glucose_conc']==0])))
print("# rows missing in diastolic_bp {0} ".format(len(df.loc[df['diastolic_bp']==0])))
print("# rows missing in Thickness {0} ".format(len(df.loc[df['thickness']==0])))
print("# rows missing in insulin {0} ".format(len(df.loc[df['insulin']==0])))
print("# rows missing inbmi {0} ".format(len(df.loc[df['bmi']==0])))
print("# rows missing in diab_pred {0} ".format(len(df.loc[df['diab_pred']==0])))
print("# rows missing in age {0} ".format(len(df.loc[df['age']==0])))


# ## Imputing with mean

# In[41]:


from sklearn.preprocessing import Imputer

fill_0 = Imputer(missing_values = 0, strategy = "mean", axis =0)
x_train = fill_0.fit_transform(x_train) ## filling those values in train data
x_test = fill_0.fit_transform(x_test) ## filling in test data


# # Training using Naive bayes

# In[42]:


from sklearn.naive_bayes import GaussianNB


# In[43]:


nb_model= GaussianNB()

nb_model.fit(x_train,y_train.ravel())


# ### performance on training data 

# In[47]:


nb_predict_train = nb_model.predict(x_train)


#importing the performance library
from sklearn import metrics

#Accuracy

print("Accuracy: {0:4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))
print()


# ### performance on testing data

# In[48]:


nb_predict_test = nb_model.predict(x_test)


#importing the performance library
from sklearn import metrics

#Accuracy

print("Accuracy: {0:4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))
print()


# # Victoryyy!!!! it's above 70%
# ## Metrics

# In[58]:


print("Confusion matrix")
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test)))

      
##print ("Classification report")
##print (metrics.clasification_report(y_test, nb_predict_test))


# # Random forest

# In[59]:


from sklearn.ensemble import RandomForestClassifier


# In[60]:


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train,y_train.ravel())


# In[61]:


rf_predict_train = rf_model.predict(x_train)
print("Accuracy: {0:4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))


# In[63]:


rf_predict_test = rf_model.predict(x_test)
print("Accuracy: {0:4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))


# ##### The accuracy difference between train model and test model is very therefore, random forest is not an optimal choice because of overfitting

# ## Logistic regression

# In[64]:


from sklearn.linear_model import LogisticRegression


# In[67]:


lr_model = LogisticRegression(C=0.7 , random_state=42)
lr_model.fit(x_train, y_train.ravel())
lr_predict_test = lr_model.predict(x_test)
print("Accuracy: {0:4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))


# In[ ]:




