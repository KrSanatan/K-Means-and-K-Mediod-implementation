import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
import seaborn as sns

#loading csv data
df = pd.read_csv('C:\\Users\\HP\\Desktop\\PYTHON_LEARN\\Country-data.csv')

'''Data Cleaning'''
#removing empty data
new_df = df.dropna()
#removing duplicates
new_df.drop_duplicates(inplace = True)


#code for k means < comment this for running k mediod >
# transform data
scaler = MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(new_df.iloc[ :,1:10]))
KMeans = KMeans(3,random_state=3)
identified_clusters = KMeans.fit_predict(scaled)
new_df['clusters']=identified_clusters
#plotting clustering data
plt.scatter(scaled.iloc[:,0],scaled.iloc[:,8],c=new_df['clusters'],cmap='rainbow')
plt.show()

#calculating silhouettte score
score = silhouette_score(scaled.iloc[:,1:10],identified_clusters)
print("SilHouette Score = "+str(score))


#code for k mediod < comment this for running k means >
# # transform data
# scaler = MinMaxScaler()
# scaled = pd.DataFrame(scaler.fit_transform(new_df.iloc[ :,1:10]))
# KMedoids = KMedoids(3,random_state=3)
# identified_clusters = KMedoids.fit_predict(scaled)
# new_df['clusters']=identified_clusters
# #plotting clustering data
# plt.scatter(scaled.iloc[:,4],scaled.iloc[:,8],c=new_df['clusters'],cmap='rainbow')
# plt.show()

# #calculating silhouettte score
# score = silhouette_score(scaled.iloc[:,1:10],identified_clusters)
# print("SilHouette Score = "+str(score))


#code for k means < comment this for running k mediod >
status=[]
developed = 0
under_developing = 0
developing = 0
for ind,i in new_df.iterrows():
    if(i[0]=='India'):
        developing=i[10]
    if(i[0]=='United States'):
        developed=i[10]
    if(i[0]=='Namibia'):
        under_developing=i[10]  

for ind,i in new_df.iterrows():
    if(i[10]==0):
        status.append('under_developing')
    if(i[10]==1):
         status.append('developed')
    if(i[10]==2):
         status.append('developing')       

new_df['Status']=status
print(new_df)
new_df.to_csv('final_result.csv')

#code for k mediod < comment this for running k means >
# status=[]
# developed = 0
# under_developing = 0
# developing = 0
# for ind,i in new_df.iterrows():
#     if(i[0]=='India'):
#         developing=i[10]
#     if(i[0]=='United States'):
#         developed=i[10]
#     if(i[0]=='Zambia'):
#         under_developing=i[10]  

# for ind,i in new_df.iterrows():
#     if(i[10]==0):
#         status.append('developed')
#     if(i[10]==2):
#          status.append('developing')
#     if(i[10]==1):
#          status.append('under_developing')       

# new_df['Status']=status
# print(new_df)
# new_df.to_csv('final_result_kmedoid.csv')


#to identify number of clusters via elbow 
'''wcss=[]
for i in range(1,10):
 kmeans = KMeans(i)
 kmeans.fit(new_df.iloc[ :,1:10])
 wcss_iter = kmeans.inertia_
 wcss.append(wcss_iter)

number_clusters = range(1,10)
plt.plot(number_clusters,wcss, marker = '*')
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()'''