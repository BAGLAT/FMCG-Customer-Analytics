#!/usr/bin/env python
# coding: utf-8

# ## ${\textbf{Libraries}}$

# In[1]:


import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle


# ## ${\textbf{Import Data}}$

# In[2]:


df_segmentation = pd.read_csv('segmentation data.csv', index_col = 0)
## Making index as column 1 (ID)


# ## ${\textbf{Explore Data}}$

# In[3]:


df_segmentation.head()


# In[4]:


df_segmentation.describe()
## 45.7% of sample population is female
## 49.65% of sample population is non single
## Average age of sample population is 36 yeards
## Average income of sample population is 120k dollards


# ## ${\textbf{Correlation Estimate}}$

# In[5]:


df_segmentation.corr()
## Pearson correlation decides the linear dependency of the variables
## Correlation ranges from -1 to +1
## Correlation of 0 means they are not linearly dependent


# In[6]:


plt.figure(figsize = (20, 10))
s = sns.heatmap(df_segmentation.corr(),
               annot = True, # retaining correlation coefficients with annot as True
               cmap = 'RdBu', # color red for negative and blue for positive
               vmin = -1, # boundary to -1 to 1
               vmax = 1) # boundary to -1 to 1
s.set_yticklabels(s.get_yticklabels(), rotation = 0, fontsize = 12)
s.set_xticklabels(s.get_xticklabels(), rotation = 90, fontsize = 12)
plt.title('Correlation Heatmap',size=(20))
plt.show()


# ## ${\textbf{Visualize Raw Data}}$

# In[7]:


plt.figure(figsize = (12, 9))
plt.scatter(df_segmentation.iloc[:, 2], df_segmentation.iloc[:, 4],c='r') ## Scatter plot between age and income
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Scatter Plot between Age and Income',size=20)


# ## ${\textbf{Standardization}}$

# In[8]:


## Scaling all features in same range [-a,+a]


# In[9]:


scaler = StandardScaler()
segmentation_std = scaler.fit_transform(df_segmentation)


# In[10]:


df12 = pd.DataFrame(segmentation_std)


# In[11]:


df12.head(10)


# ### Difference between K Means and Hierarchical clustering
# Hierarchical clustering can’t handle big data well but K Means clustering can. This is because the time complexity of K Means is linear i.e. O(n) while that of hierarchical clustering is quadratic i.e. O(n2).
# In K Means clustering, since we start with random choice of clusters, the results produced by running the algorithm multiple times might differ. While results are reproducible in Hierarchical clustering.
# K Means is found to work well when the shape of the clusters is hyper spherical (like circle in 2D, sphere in 3D).
# K Means clustering requires prior knowledge of K i.e. no. of clusters you want to divide your data into. But, you can stop at whatever number of clusters you find appropriate in hierarchical clustering by interpreting the dendrogram

# ## ${\textbf{Hierarchical Clustering}}$

# Hierarchical clustering, also known as hierarchical cluster analysis, is an algorithm that groups similar objects into 
# groups called clusters. The endpoint is a set of clusters, where each cluster is distinct from each other cluster, and 
# the objects within each cluster are broadly similar to each other

# A dendrogram is a diagram that shows the hierarchical relationship between objects. It is most commonly created as an output 
# from hierarchical clustering. The main use of a dendrogram is to work out the best way to allocate objects to clusters.

# In[12]:


df_segmentation.head(10)


# In[14]:


segmentation_std


# In[15]:


hier_clust = linkage(segmentation_std, method = 'ward')


# In[16]:


## We need to find the horizontal line on the dendogram on which to cut


# In[17]:


plt.figure(figsize = (12,9))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Observations')
plt.ylabel('Distance')
dendrogram(hier_clust,
           truncate_mode = 'level',
           p = 5, ## to see last 5 merged cluster levels
           show_leaf_counts = False,
           no_labels = True)
plt.show()


# ## K-means Clustering

# K-means++ is an inizialization algorithm that finds the best cluster feeds

# An ideal way to figure out the right number of clusters would be to calculate the Within-Cluster-Sum-of-Squares (WCSS). 
# WCSS is the sum of squares of the distances of each data point in all clusters to their respective centroids. The idea is to 
# minimise the sum.

# In[18]:


wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(segmentation_std)
    wcss.append(kmeans.inertia_)


# In[19]:


wcss


# In[20]:


plt.figure(figsize = (10,8))
plt.plot(range(1, 11), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means Clustering')
plt.show()


# Elbow Method - To determine K(number of optimal clusters) in K-means clustering

# In[21]:


kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)


# In[22]:


kmeans.fit(segmentation_std)


# ### ${\textbf{Results}}$

# In[23]:


df_segm_kmeans = df_segmentation.copy()
df_segm_kmeans['Segment K-means'] = kmeans.labels_


# In[24]:


df_segm_kmeans.head(10)


# In[26]:


df_segm_analysis = df_segm_kmeans.groupby(['Segment K-means']).mean()
df_segm_analysis


# Decoding above Table one by one cluster:
# 1. Cluster 0 : Label as Well-off
# 50% are male and 50% are female
# This segment consist of higher age people(old) with 70% non single and mostly have higher education.
# Average income of this people in this segment/cluster is 158000$
# 2. Cluster 1: Label as Fewer Opportunities
# 2/3rd are males, mostly single, average age is 36 years, with lower education and low average income
# Not live in big cities
# 3. Cluster 2: Label as Standard or average -  (Mostly females with medium education, in relationship)
# 4. Cluster 3: Label as Career Focused - Mostly mean, low education, high income, live in big medium size cities, single

# In[27]:


df_segm_kmeans.head(10)


# In[28]:


df_segm_analysis['N_Obs'] = df_segm_kmeans[['Segment K-means','Sex']].groupby(['Segment K-means']).count()


# In[29]:


df_segm_analysis['Prop_Obs'] = df_segm_analysis['N_Obs'] / df_segm_analysis['N_Obs'].sum()


# In[30]:


df_segm_analysis


# In[ ]:


df_segm_analysis.rename({0:'well-off',
                         1:'fewer-opportunities',
                         2:'standard',
                         3:'career focused'})


# In[95]:


df_segm_kmeans['Labels'] = df_segm_kmeans['Segment K-means'].map({0:'well-off', 
                                                                  1:'fewer opportunities',
                                                                  2:'standard', 
                                                                  3:'career focused'})


# In[96]:


df_segm_kmeans.head(10)


# In[97]:


x_axis = df_segm_kmeans['Age']
y_axis = df_segm_kmeans['Income']
plt.figure(figsize = (15, 12))
sns.scatterplot(x_axis, y_axis, hue = df_segm_kmeans['Labels'], palette = ['g', 'r', 'c', 'm'])
plt.title('Segmentation K-means')
plt.show()


# ### ${\textbf{PCA}}$

# ![Dimensionality%20Reduction.png](attachment:Dimensionality%20Reduction.png)

# In[33]:


pca = PCA()


# In[34]:


segmentation_std


# In[35]:


pca.fit(segmentation_std)


# In[ ]:





# In[36]:


pca.explained_variance_ratio_


# In[101]:


plt.figure(figsize = (12,9))
plt.plot(range(1,8), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')


# In[102]:


## If we choose 3 components we cover 80% of information 


# In[37]:


pca = PCA(n_components = 3)


# In[38]:


pca.fit(segmentation_std)


# ### ${\textbf{PCA Results}}$

# In[39]:


pca.components_


# In[40]:


df_pca_comp = pd.DataFrame(data = pca.components_,
                           columns = df_segmentation.columns.values,
                           index = ['Component 1', 'Component 2', 'Component 3'])
df_pca_comp


# In[41]:


plt.figure(figsize=(20,12))
sns.heatmap(df_pca_comp,
            vmin = -1, 
            vmax = 1,
            cmap = 'RdBu',
            annot = True)
plt.yticks([0, 1, 2], 
           ['Component 1', 'Component 2', 'Component 3'],
           rotation = 45,
           fontsize = 9)


# In[42]:


pca.transform(segmentation_std)


# In[43]:


scores_pca = pca.transform(segmentation_std)


# In[44]:


## New Data resulted from dimensionality reduction
scores_pca


# ### ${\textbf{K-means clustering with PCA}}$

# In[45]:


wcss = []
for i in range(1,11):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)


# In[46]:


plt.figure(figsize = (10,8))
plt.plot(range(1, 11), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means with PCA Clustering')
plt.show()


# In[47]:


kmeans_pca = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)


# In[48]:


kmeans_pca.fit(scores_pca)


# ### ${\textbf{K-means clustering with PCA Results}}$

# In[49]:


df_segmentation.head(1)


# In[50]:


df_segm_pca_kmeans = pd.concat([df_segmentation.reset_index(drop = True), pd.DataFrame(scores_pca)], axis = 1)


# In[52]:


df_segm_pca_kmeans.head(10)


# In[53]:


df_segm_pca_kmeans.columns.values[-3: ] = ['Component 1', 'Component 2', 'Component 3']


# In[54]:


df_segm_pca_kmeans.head(1)


# In[55]:


df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_


# In[56]:


df_segm_pca_kmeans


# In[57]:


df_segm_pca_kmeans_freq = df_segm_pca_kmeans.groupby(['Segment K-means PCA']).mean()
df_segm_pca_kmeans_freq


# In[58]:


df_segm_pca_kmeans_freq['N Obs'] = df_segm_pca_kmeans[['Segment K-means PCA','Sex']].groupby(['Segment K-means PCA']).count()
df_segm_pca_kmeans_freq['Prop Obs'] = df_segm_pca_kmeans_freq['N Obs'] / df_segm_pca_kmeans_freq['N Obs'].sum()
df_segm_pca_kmeans_freq = df_segm_pca_kmeans_freq.rename({0:'standard', 
                                                          1:'career focused',
                                                          2:'fewer opportunities', 
                                                          3:'well-off'})
df_segm_pca_kmeans_freq


# In[60]:


df_segm_pca_kmeans['Legend'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0:'standard', 
                                                          1:'career focused',
                                                          2:'fewer opportunities', 
                                                          3:'well-off'})


# In[62]:


df_segm_pca_kmeans.head(10)


# In[63]:


x_axis = df_segm_pca_kmeans['Component 2']
y_axis = df_segm_pca_kmeans['Component 1']
plt.figure(figsize = (10, 8))
sns.scatterplot(x_axis, y_axis, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm'])
plt.title('Clusters by PCA Components')
plt.show()


# In[64]:


x_axis_1 = df_segm_pca_kmeans['Component 3']
y_axis_1 = df_segm_pca_kmeans['Component 1']
plt.figure(figsize = (12, 9))
sns.scatterplot(x_axis_1, y_axis_1, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm'])
plt.title('Clusters by PCA Components' )
plt.show()


# In[65]:


x_axis_1 = df_segm_pca_kmeans['Component 3']
y_axis_1 = df_segm_pca_kmeans['Component 2']
plt.figure(figsize = (12, 9))
sns.scatterplot(x_axis_1, y_axis_1, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm'])
plt.title('Clusters by PCA Components' )
plt.show()


# ### ${\textbf{Data Export}}$

# In[131]:


pickle.dump(scaler, open('scaler.pickle', 'wb'))


# In[132]:


pickle.dump(pca, open('pca.pickle', 'wb'))


# In[133]:


pickle.dump(kmeans_pca, open('kmeans_pca.pickle', 'wb'))


# In[ ]:




