
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn.metrics
import scipy.sparse as sc
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


# In[2]:


#data = pd.read_csv('train.dat', header=None)
with open("train.dat", "r", encoding="utf8") as fh:
    lines = fh.readlines() 
    
docs = [l.split() for l in lines]
docs = [list(map(int, x)) for x in docs]


# In[3]:


##Parse Strings for each document. Even #s are the Indices, Odd #s are the Values, Indptr is the index in values when new rows start 


# In[4]:


indptr = [0]
indices = []
values = []
nrows = len(docs)

for line in docs:
    for x, val in enumerate(line):
        if x % 2 == 0:
            ##even
            indices.append(val)
        if x % 2 != 0:
            ##odd
            values.append(val)
    indptr.append(len(values))
    
ncols = len(indices)
print(len(values))
print(len(indices))
print(len(indptr))


# In[5]:


##Generate CSR Matrix
mat = csr_matrix((values, indices, indptr), shape=(nrows, ncols), dtype=np.double)
mat.sort_indices()


# In[6]:


##Dimensionality Reduction (Trial 1, Truncated SVD)


# In[7]:


dim = TruncatedSVD(n_components=200)
mat2 = dim.fit_transform(mat)
del mat
mat2 = csr_matrix(mat2)


# In[8]:


##Code from Activity-Data-3
##L2 Normalization


# In[9]:


def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat
    
mat_normalized = csr_l2normalize(mat2, copy=True)


# In[10]:


from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(mat_normalized)

#mat = mat_normalized.toarray()
#del mat_normalized
# In[11]:


###Different Cosine Sim code

#dots = mat_normalized.dot(mat_normalized.T)
#from scipy.sparse.linalg import norm
#norms = norm(mat_normalized, axis=1)
#dot2 = dots.toarray()

#for x in range(dots.shape[0]):
#    for y in range(dots.shape[0]):
#        dot2[x][y] = dot2[x][y]/(norms[x]*norms[y])


# In[12]:


import math 
def dbscan(matrix, eps, minpts):
    cluster = 1
    labels = [np.nan]*len(matrix)
    typept = [np.nan]*len(matrix)
    ##use similarity matrix
    x = 0
    for d in matrix:
        pts = [] #indexes of documents in sim matrix
        y = 0
        val = 0
        for y, val in enumerate(d):
            #print("y", y)
            #print("val", val)
            if ((val >= (1-eps)) & (val < 1)): ##find all closest points to d w/in eps
                pts.append(y)
                #print("yay")
        if len(pts) == 0:
            labels[x] = cluster
            typept[x] = 'Noise'
            cluster += 1
        else:
            isborder = False
            if len(pts) < minpts:
                for p in pts:
                    if p == 'Core': isborder = True
                if isborder: typept[x] = 'Border'
                else: typept[x] = 'Noise'
            else: typept[x] = 'Core'
        ##check for labels in labels array
            if not math.isnan(labels[x]):
                #set un-labelled pts in labels[pts] to same value as center
                for pt in pts:
                    #print(pt)
                    if math.isnan(labels[pt]):
                        #print(labels[pt], "before")
                        labels[pt] = labels[x]
                        #print(labels[pt], "after")
            else:
                ##if all pts in labels[pts] are null, set as new cluster
                allnull = True
                lab = []
                typ = []
                notlab = []
                for pt in pts:
                    if not math.isnan(labels[pt]):
                        lab.append(labels[pt]) ##labels of pts that are labelled
                        allnull = False
                    else:
                        notlab.append(pt) ##pts that are not labelled
                    if typept[pt] == 'Core':
                        typ.append(pt) #neighbors that have a type specified as Core
                if allnull:
                    labels[x] = cluster
                    for pt in pts:
                        labels[pt] = labels[x]
                    cluster += 1
                else:
                    if len(typ) == 0:                    #label[x] = statistics.mode(lab)
                        labels[x] = max(lab, key = lab.count)
                        for p in notlab:
                            labels[p] = labels[x]
                    else:
                        labels[x] = labels[typ[0]]
                        for p in notlab:
                            labels[p] = labels[x]                  

        x += 1
        #print("d", d)
    #print(cluster)
    #print(len(labels))
    #with open('types.txt', 'w') as f:
     #   for item in typept:
      #      f.write("%s\n" % item)
    return labels, cluster


# In[13]:


label, cluster = dbscan(similarities, 0.25, 5)
print(cluster)
del similarities
del docs
# In[14]:


with open('retest.txt', 'w') as f:
    for item in label:
        f.write("%s\n" % item)

# In[ ]:
