#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix

X_training = df.to_numpy()
maxk = 0
max = 0
k_values = []
s_values = []
#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
for k in range(2,21):
     kmeans = KMeans(n_clusters = k, random_state=0)
     kmeans.fit(X_training)



     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
     kscore = silhouette_score(X_training, kmeans.labels_)
     if kscore > max:
          max = kscore
          maxk = k
     k_values.append(k)
     s_values.append(kscore)
print(maxk)
#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here

plt.plot(k_values, s_values)
plt.ylabel('Silhouette Score')
plt.xlabel('K value')
plt.show()

#reading the validation data (clusters) by using Pandas library
#--> add your Python code here
df_test = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here

labels = np.array(df_test.values).reshape(1,df_test.size)[0]
#Calculate and print the Homogeneity of this kmeans clustering
kmeans = KMeans(n_clusters = maxk, random_state=0)
kmeans.fit(X_training)
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here


#rung agglomerative clustering now by using the best value o k calculated before by kmeans
#Do it:
#agg = AgglomerativeClustering(n_clusters=<best k value>, linkage='ward')
#agg.fit(X_training)
agg = AgglomerativeClustering(n_clusters= maxk, linkage='ward')
agg.fit(X_training)

#Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
