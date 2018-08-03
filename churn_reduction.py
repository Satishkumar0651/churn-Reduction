
# coding: utf-8

# # Python code for Churn Reduction

# In[ ]:

# Load libraries
import os
import pandas as pd
import numpy as np
from fancyimpute import KNN
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform


# In[2]:

#Setting path
os.chdir("E:/1ST SEM/eng/edwisor_assignments/8.project 2")


# # Loading data

# In[3]:

#loading train and test data
df_train = pd.read_csv("Train_data.csv")
df_test = pd.read_csv("Test_data.csv")


# In[4]:

cnames = ["account length","area code","number vmail messages","total day minutes","total day calls","total day charge",
"total eve minutes","total eve calls","total eve charge","total night minutes","total night calls",
"total night charge","total intl minutes","total intl calls", "total intl charge",
"number customer service calls"]


# # Outlier Analysis

# In[5]:

d1=df_train[cnames]
d2=df_test[cnames]


# In[ ]:




# In[6]:

for i in cnames:
    #Detect and replace with NA
    #Extract quartiles
    q75, q25 = np.percentile(d1.loc[:,i], [75 ,25])

    #Calculate IQR
    iqr = q75 - q25

    #Calculate inner and outer fence
    mini = q25 - (iqr*1.5)
    maxi = q75 + (iqr*1.5)

    #Replace with NA
    d1.loc[d1.loc[:,i] < mini,:i] = np.nan
    d1.loc[d1.loc[:,i] > maxi,:i] = np.nan
    # df_train["state"]=NULL
    #impute with median
    #d1[:,i]=d1[:,i].fillna(d1[:,i].median())


# In[7]:

for i in cnames:
    #Detect and replace with NA
    #Extract quartiles
    q75, q25 = np.percentile(d2.loc[:,i], [75 ,25])

    #Calculate IQR
    iqr = q75 - q25

    #Calculate inner and outer fence
    mini = q25 - (iqr*1.5)
    maxi = q75 + (iqr*1.5)

    #Replace with NA
    d2.loc[d2.loc[:,i] < mini,:i] = np.nan
    d2.loc[d2.loc[:,i] > maxi,:i] = np.nan
    # df_train["state"]=NULL
    #impute with median
    #d1[:,i]=d1[:,i].fillna(d1[:,i].median())


# ### MIssing value imputation after outlier analysis

# In[8]:

d1=d1.apply(lambda x:x.fillna(x.median()),axis=0)
d2=d2.apply(lambda x:x.fillna(x.median()),axis=0)


# In[9]:

d1["international plan"]=df_train["international plan"]
d1["voice mail plan"]=df_train["voice mail plan"]
d1["churn"]=df_train["Churn"]

d2["international plan"]=df_test["international plan"]
d2["voice mail plan"]=df_test["voice mail plan"]
d2["churn"]=df_test["Churn"]


# In[10]:

#assigning levels to categorical varibales
for i in range(0, d1.shape[1]):
    if(d1.iloc[:,i].dtypes == 'object'):
        d1.iloc[:,i] = pd.Categorical(d1.iloc[:,i])
        d1.iloc[:,i] = d1.iloc[:,i].cat.codes

for i in range(0, d2.shape[1]):
    if(d2.iloc[:,i].dtypes == 'object'):
        d2.iloc[:,i] = pd.Categorical(d2.iloc[:,i])
        d2.iloc[:,i] = d2.iloc[:,i].cat.codes

d2.head(10)


# In[11]:

#storing target variable
train_targets = d1.churn
test_targets = d2.churn


# In[12]:

#combining train and test data for data prepocessing
combined = d1.append(d2)

print(combined.shape, d1.shape, d2.shape)


# In[13]:

combined.head()


# ## correlation plot

# In[14]:

df_corr = combined.loc[:,cnames]
get_ipython().run_line_magic('matplotlib', 'inline')
#correlation analysis
#set height and width of plot
f , ax = plt.subplots(figsize = (15,12))
#generate correlation matrix
corr = df_corr.corr()
#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap='rainbow',annot=True,
            square=True, ax=ax)


# In[15]:

cat_names = ["international plan","voice mail plan"]

#chi square test of independence
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(combined["churn"],combined[i]))
    print(p)


# In[16]:

combined.info()


# ## Dimensionality Reduction

# In[17]:

#dropping unnecessary variables
combined = combined.drop(["total day minutes", "total eve minutes", "total night minutes", "total intl minutes"], axis = 1)

combined.shape


# In[18]:

cnames = ["account length","area code","number vmail messages","total day calls","total day charge",
"total eve calls","total eve charge","total night calls","total night charge","total intl calls",
"total intl charge", "number customer service calls"]


# ## Normalization

# In[19]:

#normalization
for i in cnames:
    print(i)
    combined[i] = (combined[i]-min(combined[i]))/(max(combined[i])-min(combined[i]))

combined.head(10)


# In[20]:

#loading libraries for model
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
#splitting combined data to train and test
train = combined[:3333]

test = combined[3333:]


# In[21]:

features=["account length","area code","number vmail messages","total day calls","total day charge",
"total eve calls","total eve charge","total night calls","total night charge","total intl calls", "total intl charge",
"number customer service calls","international plan","voice mail plan"]


# In[22]:

X=combined[features]


# In[23]:

y=combined.churn


# ## Decision Tree model

# In[24]:

#decision tree model
c50_model = tree.DecisionTreeClassifier(criterion = 'entropy').fit(train, train_targets)
c50_pred = c50_model.predict(test)

c50_pred

#dot file to look at decision tree
dotfile = open("pt.dot", 'w')
df = tree.export_graphviz(c50_model, out_file=dotfile, feature_names = train.columns)

#testing accuracy of model
from sklearn.metrics import confusion_matrix
CM = pd.crosstab(test_targets, c50_pred)
CM

TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
accuracy_score(test_targets,c50_pred)*100
#accuracy = 92.32153
#(FN*100)/(FN+TP)
#FNR = 32.142857
#(TP*100)/(TP+FN)
#Recall = 67.8571428


# ## Cross Validation

# In[25]:

# 10-fold cross-validation with logistic regression
print(cross_val_score(c50_model, X, y, cv=10, scoring='accuracy').mean())


# In[26]:

accuracy_score(test_targets,c50_pred)*100


# In[27]:

from sklearn.model_selection import cross_val_score


# ## Random Forest model

# In[28]:

#random forest model
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 100).fit(train,train_targets)
RF_prediction = RF_model.predict(test)


RF_prediction

CM = pd.crosstab(test_targets, RF_prediction)
CM

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
#check accuracy of model
accuracy_score(test_targets, RF_prediction)*100
#((TP+TN)*100)/(TP+TN+FP+FN)
#accuracy = 94.96100
#(FN*100)/(FN+TP)
#FNR = 33.928571


# In[29]:

# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
scores = cross_val_score(RF_model, X, y, cv=10, scoring='accuracy')
print(scores)


# ## KNN Model

# In[ ]:

#KNN implementation
from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors = 9).fit(train, train_targets)

#predict test cases
KNN_Predictions = KNN_model.predict(test)

#build confusion matrix
CM = pd.crosstab(test_targets, KNN_Predictions)
CM

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)
#accuracy = 86.622675
#False Negative rate
#(FN*100)/(FN+TP)
#FNR =97.321428



# In[31]:

from sklearn.model_selection import cross_val_score


# In[32]:

# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
scores = cross_val_score(KNN_model, X, y, cv=10, scoring='accuracy')
print(scores)


# ## Naive Bayes

# In[33]:

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
#Naive Bayes implementation
NB_model = GaussianNB().fit(train, train_targets)

#predict test cases
NB_Predictions = NB_model.predict(test)

#Build confusion matrix
CM = pd.crosstab(test_targets, NB_Predictions)
CM

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
#check accuracy of model
#accuracy_score(Y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)
#Accuracy = 85.842831433
#False Negative rate
#(FN*100)/(FN+TP)
#FNR = 60.2678571

#we will be fixing random forest model as it provides best results
#now we will generate example out for out sample input test data with Random forest predictions
move = pd.DataFrame(RF_prediction)
move = move.rename(columns = {0:'move'})

test = test.join(move['move'])

test.to_csv("example_output.csv", index = False)


# In[ ]:

# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
scores = cross_val_score(NB_model, X, y, cv=10, scoring='accuracy')
print(scores)

