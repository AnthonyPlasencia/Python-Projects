#Anthony Plasencia
#Comp 541
#Import necessary libraries (e.g., pandas, numpy).

import pandas as pd
import numpy as nu 
import csv
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#new lines
def newLine():
    print("--------------------------------------------------------------------------------------------------------------------------------------")

# #Load the dataset using pandas (e.g., pd.read_csv()).
data = pd.read_csv('College.csv')

# Display the first few rows of the dataset using head() to show an overview
# print(data.head()) shows the first 5 rows of the dataset and the last 5 rows
print(data.head())

newLine()

# Use info() and describe() to provide a summary of the dataset, including data types and basic statistics
# print(data.info()) shows 4 columns and 18 rows, it shows all the data types and the name of each column and the number of non-null values
# print(data.describe()) gives count, mean, std, min, 25%, 50%, 75%, max for each column in the dataset
print("The Info function")
print(data.info())
newLine()

print("The Describe function")
print(data.describe())
newLine()

#data.info() shows that there are no null values so there is no need to adress missing values
#if there were missing values I would use data.dropna() to drop the rows with missing values

#Convert categorical data into a numerical format (e.g., using pd.get_dummies() for one-hot encoding).
#I changed split the Private column into two columns one for yes and one for no and changed the prefix to make it easier to read
dummy = pd.get_dummies(data,columns=['Private'], prefix="Private", dtype=float)
print("The pd.get_dummies() for the Private column and before the normalizing the data:")
print(dummy.head())
newLine()

#Normalize the data: Apply normalization (e.g., MinMaxScaler or StandardScaler from sklearn) and show before-and-after statistics to illustrate the effect.
print("The Normalized Data using StandardScaler:")
scaler = StandardScaler()
#changed the dataset to exclude the school name
NoSchoolName = dummy.iloc[:,1:]
Normalized = pd.DataFrame(scaler.fit_transform(NoSchoolName), columns=NoSchoolName.columns)
print(Normalized.head())
newLine()

plt.rcParams["figure.figsize"] = (18, 10)
#Visualizations: Include plots like histograms, boxplots, and scatter plots to explore data distributions and relationships.
#Shows scatter plot of the PhD and Grad.Rate columns
Normalized.plot(kind='scatter', x='Accept', y='Grad.Rate')
plt.title('Scatter plot of Accept and Grad.Rate')
plt.show()
# the scatter plot doesn't show a strong correlation between the two columns

#Show histogtram
Normalized.hist()
plt.title('Histogram of the Normalized Data')
plt.show()
#The histogram shows the distribution of each column in the dataset
#11/16 columns are right skewed while 2/16 are left skewed and 3/16 are normal
#I ignored the Private_yes and Private_no columns because they are binary

#Shows Boxplot
plt.boxplot(Normalized,)
plt.title('Boxplot of the Normalized Data')
plt.show()
#The Boxplt shows the outliers the quartiles and the median of the dataset
#I can see there are outliers in the dataset but other than that the graphs are diffucult to read

#Shows heatmap and the correlation between the columns
annot = True
sn.heatmap(Normalized.corr(), annot=annot)
plt.title('Heatmap of the Normalized Data')
plt.show()
#The Heatmap shows the correlation between the columns in the dataset the colors show the strength of the correlation
#most of the columns are not strongly correlated with each other at a glance
#But there are some columns that are correlated with each other like F.Undergrad and Enroll and Private_yes


#Summary statistics: Compute and display key statistics (mean, median, standard deviation, etc.).
print("The Statitics of the Normalized Data:")
print(Normalized.describe())
newLine()

#Import K-means from sklearn and apply it to the dataset.
#Show the process of determining the optimal number of clusters (e.g., using the elbow method plot).
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
print("Results of the Elbow Method:")
K = range(1,10)
for k in K:
    KMeansModel = KMeans(n_clusters=k).fit(Normalized)
    KMeansModel.fit(Normalized)

    distortions.append(sum(nu.min(cdist(Normalized,KMeansModel.cluster_centers_,'euclidean'), axis=1)) / Normalized.shape[0])
    inertias.append(KMeansModel.inertia_)

    mapping1[k] = sum(nu.min(cdist(Normalized,KMeansModel.cluster_centers_,'euclidean'), axis=1)) / Normalized.shape[0]
    mapping2[k] = KMeansModel.inertia_
for key, val in mapping1.items():
    print(f'{key} : {val}')
#Plot the elbow method


plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()
#The plot of the elbow method shows that the optimal number of clusters is 5 since the slope of the graph starts to decrease after 5 rapidly

#Display the final clustering results, including cluster centers and labels for each data point.
k_range = range(1,5)

inertiaValues = []

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(Normalized)
inertiaValues.append(kmeans.inertia_)
plt.scatter(Normalized.iloc[:, 0], Normalized.iloc[:, 1], c=y_kmeans)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')
plt.title('K-means clustering (k={})'.format(5))
plt.show()

newLine()

#Evaluate the clustering results using silhouette score.
print("The Silhouette Score of the KMeans Clustering:")
kmeans = KMeans(n_clusters=5, random_state=42)
print(silhouette_score(Normalized, kmeans.fit_predict(Normalized)))
newLine()

#Evaluate the clustering results within-cluster sum of squares.
print("The Within-Cluster Sum of Squares of the KMeans Clustering:")
kmeans = KMeans(n_clusters=3, random_state=0).fit(Normalized)
wcss = kmeans.inertia_
print('WCSS: ', wcss)
newLine()

#Use PCA or other techniques to reduce the dimensions of the data.
print("The PCA of the Normalized Data:")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(Normalized)
print(X_pca)

#Plot the clusters in this reduced-dimensional space esure each cluster is clearly visualized.
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X_pca)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')
plt.title('K-means clustering (k={})'.format(5))
plt.show()
newLine()

#The custering results show that the data in the yellow and purple clusters are super close to eachother 
#compared to the other clusters, Not only are they close the data points are also very close to the cluster center
#this shows that the schools who are appart of the yellow and purple clusters are very similar to eachother
#The Data in the blue green and light blue clusters are all very spread out and are not close to the cluster center
#The Green cluster is the most spread out and the data points are the farthest from the cluster center
#This shows that the schools in thhese clusters are not thatt similar to eachother

#I think this shows that in the bigger picture US Colleges at least based on the information from the data set shows that some
#schools are very similar to eachother while most others are not similar at all so when deciding on a college to go to
#It can be hard to find a college that is similar to another college