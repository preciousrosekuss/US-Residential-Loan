#!/usr/bin/env python
# coding: utf-8

# In[2]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


get_ipython().system('wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv')


# In[4]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# In[7]:


df['loan_status'].value_counts()


# In[8]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[9]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[10]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[11]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# In[12]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# In[13]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[14]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# In[15]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# In[16]:


df[['Principal','terms','age','Gender','education']].head()


# In[17]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# In[18]:


X = Feature
X[0:5]


# In[19]:


y = df['loan_status'].values
y[0:5]


# In[20]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)


# In[37]:


from sklearn.neighbors import KNeighborsClassifier
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)


# In[44]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    y_kmeans_predicted=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, y_kmeans_predicted)
mean_acc


# In[45]:


k = 7
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)


# In[46]:


y_kmeans_predicted=neigh.predict(X_test)


# In[47]:


from sklearn.tree import DecisionTreeClassifier
clftree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
clftree.fit(X_train,y_train)
predTree = clftree.predict(X_test)


# In[48]:


from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))


# In[49]:


from sklearn import svm
svM = svm.SVC(kernel='rbf')
svM.fit(X_train, y_train) 

yhat = svM.predict(X_test)


# In[50]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools

from sklearn.metrics import f1_score


# In[51]:


f1_score(y_test, yhat, average='weighted') 


# In[52]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)


# In[53]:


yhat = LR.predict(X_test)


# In[54]:


from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,average=None)


# In[55]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# In[56]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# In[57]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[58]:


test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_Feature = test_df[['Principal','terms','age','Gender','weekend']]
test_Feature = pd.concat([test_Feature,pd.get_dummies(test_df['education'])], axis=1)
test_Feature.drop(['Master or Above'], axis = 1,inplace=True)
test_X = preprocessing.StandardScaler().fit(test_Feature).transform(test_Feature)


# In[59]:


new_X_test=test_X
new_y_test = test_df['loan_status'].values


# In[60]:


new_y_test_kmeans_predicted=neigh.predict(new_X_test)


# In[61]:


a=jaccard_score(new_y_test,new_y_test_kmeans_predicted,average=None).max()


# In[62]:


b=f1_score(new_y_test,new_y_test_kmeans_predicted, average='weighted') 


# In[63]:


new_y_test_Dtree_predicted=clftree.predict(new_X_test)


# In[64]:


c=jaccard_score(new_y_test,new_y_test_Dtree_predicted, average=None).max()


# In[65]:


d=f1_score(new_y_test,new_y_test_Dtree_predicted, average='weighted') 


# In[67]:


new_y_test_svm_predicted=svM.predict(new_X_test)


# In[68]:


e=jaccard_score(new_y_test,new_y_test_Dtree_predicted, average=None).max()


# In[69]:


f=f1_score(new_y_test,new_y_test_Dtree_predicted, average='weighted') 


# In[70]:


new_y_test_LR_predicted=LR.predict(new_X_test)
g=jaccard_score(new_y_test,new_y_test_LR_predicted, average=None).max()


# In[71]:


h=f1_score(new_y_test,new_y_test_LR_predicted, average='weighted') 


# In[72]:


LR_yhat_prob = LR.predict_proba(new_X_test)

i=log_loss(new_y_test,LR_yhat_prob)


# In[73]:


import pandas as pd  
  
# assign data of lists.  
Report = {'Algorithm': ['KNN', 'Decision Tree', ' SVM', 'LogisticRegression'], 'Jaccard': [a , c, e,g],'F1-score': [b, d, f,h],'LogLoss': ['NA', 'NA', 'NA', i]}  
  
# Create DataFrame  
df = pd.DataFrame(Report)  
  
# Print the output.  
df


# In[ ]:




