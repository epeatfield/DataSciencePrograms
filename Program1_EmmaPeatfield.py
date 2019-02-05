
# coding: utf-8

# In[1]:


import numpy as np
import re
import string
from nltk.stem.lancaster import LancasterStemmer
import math
from scipy.sparse import csr_matrix
#from statistics import mode


#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords #stack overflow
from nltk.tokenize import word_tokenize #geeks for geeks



#splitting the data into classes and text
cls = []
data = []
with open('train.dat') as file:
    content = file.readlines()
    for lines in content:
        lines = lines.strip('\n')
        lines = lines.strip('\t')
        cls.append(lines[0])
        data.append(lines[2:])

def preprocess(data):
    #removing numbers, via stackoverflow https://stackoverflow.com/questions/30315035/strip-numbers-from-string-in-python
    output = []
    i = 0
    for entry in data:
        output = re.sub(r'\d+', '', entry)
        data[i] = output
        i += 1

    #removing functional words
    stop_words = set(stopwords.words('english'))

    i = 0
    for entry in data:
        word_tokens = word_tokenize(entry.lower())
        output = ''
        for w in word_tokens:
            if w not in stop_words:
                output = output + ' ' + w
        data[i] = output
        i += 1
    #removing punctuation and other symbols
    i = 0
    for entry in data:
        data[i] = entry.translate(str.maketrans('','',string.punctuation)) #stackover flow
        i += 1

    #stemming/standardization
    i = 0
    for entry in data:
        ps = LancasterStemmer()
        word_tokens = word_tokenize(entry)
        output = ''
        for w in word_tokens:
            w = ps.stem(w)
            output = output + ' ' + w
        data[i] = output
        i += 1

preprocess(data)


# In[2]:


numtraindoc = len(data)

docs = []
for i in range(0, numtraindoc):
    docs.append(data[i].split())

indptr = [0]
indices = []
traindata = []
vocabulary = {}
###Creating csr_matrix pf term frequency in the training data from scipy documentation
for d in docs:
    for term in d:
        index = vocabulary.setdefault(term, len(vocabulary))
        indices.append(index)
        traindata.append(1)
    indptr.append(len(indices))

wmatrix = csr_matrix((traindata, indices, indptr), dtype=float).toarray()
length = len(wmatrix)
print(length)


##computes the tf-idf for the documents
j = 0
for entry in wmatrix:
    lengthdoc = len(docs[j])
    sumcol = 0
    for i in range(0, length):
        if wmatrix[i,j] != 0: sumcol += 1
    idf = 1 + math.log(length/sumcol, 10)
    wmatrix[:,j] *= idf
    wmatrix[j,:] /= lengthdoc
    j += 1


# In[3]:


#intake test and preprocess
pred = []

with open('test.dat') as file:
    content = file.readlines()
    for lines in content:
        pred.append(lines)

preprocess(pred)
lengpred = len(pred)

print("Length of Test Data:", lengpred)

testdocs = []
for i in range(0, lengpred):
    testdocs.append(pred[i].split())


indptr2 = [0]
indices2 = []
testdata = []
###Creating csr_matrix pf term frequency in the test data from scipy documentation

for d in testdocs:
    for term in d:
        if term in vocabulary:
            index = vocabulary.setdefault(term, len(vocabulary))
            indices2.append(index)
            testdata.append(1)
    indptr2.append(len(indices2))

testmatrix = csr_matrix((testdata, indices2, indptr2), shape = (lengpred, len(vocabulary)), dtype=float).toarray()
length = len(testmatrix)
print(len(testmatrix[1]))
print(len(wmatrix[1]))
print(len(vocabulary))
#print(testdata)

##finding tf-idf for test documentation
j = 0
for entry in testmatrix:
    lengthdoc = len(testdocs[j])
    sumcol = 0
    for i in range(0, length):
        if testmatrix[i,j] != 0: sumcol += 1
    if sumcol != 0:
        idf = 1 + math.log(lengpred/sumcol, 10)
    testmatrix[:,j] *= idf
    testmatrix[j,:] /= lengthdoc
    j += 1


# In[7]:


#FINDING LENGTH OF EACH DOCUMNT AND SAVING IN ARRAY FOR LATER COSINE SIM CALCULATION
testv = [0]*lengpred
trainv = [0]*numtraindoc
i = 0
x = []
x = testmatrix**2
sm = x.sum(axis=1, dtype=float) # sum the rows
testv = np.sqrt(sm)



y = []
y = wmatrix**2
sm = y.sum(axis=1, dtype=float)
trainv = np.sqrt(sm)


# In[16]:


## CALCULATE COSINE SIM BETWEEN EACH TEST AND TRAINING DOCUMNT
##TAKES A WHILE, SO BE PATIENT (Working on optimization)

finalsim = [[0] * numtraindoc for i in range(lengpred)]
test = 0
train = 0
x = []
for i in range(0, lengpred):
    dots = []
    x = testmatrix[i]
    dots = wmatrix.dot(x)
    test = testv[i]
    for k in range(0, numtraindoc):
        finalsim[i][k] = (dots[k]/(test*trainv[k]))

# In[17]:


#For each test doc, find the index of the k highest similarity measures and find their associated classes
#Then suggest what the prediction of document should be based on k neighbors classes

def kmax(final, k, cls):
    for i in range(0, len(final)):
        maxvals = []
        for l in range(0, k):
            ind = np.argmax(final[i])
            maxvals.append(ind)
            final[i][ind] = 0
        clsfinal = []
        for j in range(0, len(maxvals)):
            clsfinal.append(int(cls[maxvals[j]]))

        with open('results.txt', 'a') as f:
                print(max(clsfinal, key = clsfinal.count), file=f)
    print("Done")

kmax(finalsim, 10, cls)
