
# coding: utf-8

# In[22]:


import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
scaler = StandardScaler()


#import data
data = np.loadtxt('train.dat', dtype=np.float)

#import test data
test_data = np.loadtxt('test.dat', dtype=np.float)


# In[23]:


#import labels
cls = []
with open('train.labels') as file:
    content = file.readlines()
    for lines in content:
        cls.append(lines[0])

#x, testx, y, testy = train_test_split(data, cls, test_size=0.3) # 70% training and 30% test


# In[25]:
###Feature Selection
fs = VarianceThreshold()
fs.fit_transform(data)
fs.fit_transform(test_data)

###FACTOR ANALYSIS
        
#FA = FactorAnalysis(n_components = 3).fit_transform(data, cls)


###RANDOM FOREST

#X_RFtrain, X_RFtest, y_RFtrain, y_RFtest = train_test_split(data, cls, test_size=0.3) # 70% training and 30% test

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=5)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(data, cls)

#y_RFpred=clf.predict(X_RFtest)


###PRINCIPAL COMPONENT ANALYSIS"


# Apply transform to both the training set and the test set.
#X = scaler.fit_transform(data) #only fit the "training" data
#X_test = scaler.transform(test_data) 

##Real Code
#training = scaler.fit_transform(data)
#labels = cls[:]
#test = scaler.transform(test_data)


#pca = PCA(0.95) #find out the different parameters for this
#pca.fit(X_train)

#pca.fit(training) 

#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)

#training = pca.transform(training)
#test = pca.transform(test)

print("Calculating Sim")
# In[30]:


##CLASSIFICATION TIME


# In[31]:


#NEAREST NEIGHBOR

#similarity calculation (pearson/cosine similarity)
def similar(train, test):
    num_trainim = len(train)
    t1 = [0]*num_trainim
    x = []
    x = train**2
    sm = x.sum(axis=1, dtype=float)
    t1 = np.sqrt(sm)

    num_testim = len(test)
    t2 = [0]*num_testim
    y = []
    y = test**2
    sm = y.sum(axis=1, dtype=float)
    t2 = np.sqrt(sm)


    sim = [[0] * num_trainim for i in range(num_testim)]
    x = []
    for i in range(0, num_testim):
        dots = []
        x = test[i]
        t_test = t2[i]
        for j in range(0,num_trainim):
            if i == j:
                dots.append(0)
            if i != j:
                dots.append(train[j].dot(x))
                t_train = t1[j]
                sim[i][j] = (dots[j]/(t_train*t_test))
                
    return sim


#simPCA = []
#simRF = []

#simPCA = similar(training, test)
#simRF = similar(data, test_data)

print("Finding Nearest Neighbor")
#find nearest neighbors
def findkneighbs(sim, labelstrain, k):
    y_pred = []
    for i in range(0, len(sim)):
        maxvals = []
        for l in range(0, k):
            ind = np.argmax(sim[i])
            maxvals.append(ind)
            sim[i][ind] = 0
        clsfinal = []
        for j in range(0, len(maxvals)):
            clsfinal.append(labelstrain[maxvals[j]])
        
        y_pred.append(max(clsfinal, key = clsfinal.count))
        with open('results.txt', 'a') as f:
                print(max(clsfinal, key = clsfinal.count), file=f)
    print("Done")
    return y_pred


# In[36]:


#y_pred = findkneighbs(simPCA, labels, 5)
#y_rfpred = findkneighbs(simRF, cls, 5)


###RF and KNN from sklearn
#nRF = KNeighborsClassifier(n_neighbors=5)
#nRF.fit(data, cls)
#y_pred = nRF.predict(test_data)

#from sklearn.neighbors.nearest_centroid import NearestCentroid
#nnc = NearestCentroid()
#nnc.fit(data, cls)
#y_pred = nnc.predict(test_data)
    
y_pred = clf.predict(test_data)

with open('results.txt', 'a') as f:
    for x in range(len(y_pred)):
        print(y_pred[x], file=f)
# In[ ]:


#NAIVE BAYES
#gnb = GaussianNB()
#y_pred = gnb.fit(data, cls).predict(test_data)
#mnb = MultinomialNB()
#y_pred2 = mnb.fit(data, cls).predict(test_data)
#with open('results.txt', 'a') as f:
#    for x in range(len(y_pred2)):
#        print(y_pred2[x], file=f)
# In[ ]:


##METRICS


# In[37]:


#print(confusion_matrix(y_test,y_pred)) 
#print("PCA INFORMATION")
#print(classification_report(y_test,y_pred))  
#print(accuracy_score(y_test, y_pred)) 
#print(f1_score(y_test, y_pred, average='weighted'))


# In[38]:


#print("RF INFORMATION")
#print(classification_report(y_RFtest,y_rfpred))  
#print(accuracy_score(y_RFtest, y_rfpred)) 
#print(f1_score(y_RFtest, y_rfpred, average='weighted'))
