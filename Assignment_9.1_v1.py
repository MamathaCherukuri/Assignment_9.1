#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the iris dataset
from sklearn.datasets import load_iris


# In[2]:


iris=load_iris()


# In[3]:


import pandas as pd


# In[4]:


import numpy as np


# In[5]:


data=pd.DataFrame(iris.data,columns=iris.feature_names)


# In[6]:


label=pd.DataFrame(list(map(lambda x: iris.target_names[x],iris.target)),
                   columns=['Species'])


# In[7]:


iris=pd.concat([data,label],axis=1)


# In[40]:


print(iris.head(20))


# In[ ]:


Use the distplot() to see the distribution of the sepallength cm,
Sepalwidth cm, PetalLength Cm features. Plot them as subplots in a single image.
  


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


sns.set(font_scale=1.5,palette='colorblind',style='white')
g=sns.pairplot(iris,hue='Species',size=2,aspect=1)
g.fig.suptitle('Pair plot of flower dimensions split by Species')


# In[ ]:


Do a bloxplot of all features except "Species"


# In[15]:


iris=sns.load_dataset("iris")
ax=sns.boxplot(data=iris,orient="h",palette="Set2")


# In[23]:


#Do a countplot for the feature species
iris=sns.load_dataset("iris")
sns.set(style="darkgrid")
sns.countplot(x="species",data=iris)


# In[ ]:


#Do a pairplot on the features "SepalLengthCm", "SepalWidthCm"
#,"PetalLengthCm","PetalWidthCm,"Species")


# In[26]:


g=sns.PairGrid(iris,hue="species",hue_kws={"cmap":["Blues","Greens","Reds"]})
g=g.map_diag(sns.kdeplot,lw=3)
g=g.map_offdiag(sns.kdeplot,lw=1)

plt.show()


# In[ ]:


#Do an lmplot on the following SepalLengthCm, PetalLengthCm. Using hue, display the 
# different species in different colors.


# In[37]:



sns.lmplot("sepal_length", "sepal_width", data=iris, hue="species", fit_reg=False)


# In[34]:


iris.dtypes


# In[ ]:


#Do a barplot of species vs sepallength


# In[38]:


sns.barplot(x="species",y="sepal_length",data=iris)


# In[ ]:


#using heatmap, plot correlation matrix


# In[45]:


corr=iris.corr()
sns.heatmap(corr,linewidth=0.3,vmax=1.0,square=True,annot=True)


# In[ ]:





# In[ ]:





# In[ ]:




