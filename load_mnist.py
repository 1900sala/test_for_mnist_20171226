
# coding: utf-8

# In[19]:


import numpy as np
import struct
from sklearn import preprocessing

def loadImageSet(filename):

	binfile= open(filename, 'rb')
	buffers = binfile.read()
 
	head = struct.unpack_from('>IIII' , buffers ,0)

 
	offset = struct.calcsize('>IIII')
	imgNum = head[1]
	width = head[2]
	height = head[3]
	#[60000]*28*28
	bits = imgNum * width * height
	bitsString = '>' + str(bits) + 'B' #like '>47040000B'
 
	imgs = struct.unpack_from(bitsString,buffers,offset)
 
	binfile.close()
	imgs = np.reshape(imgs,[imgNum,width*height])
	return imgs
 
def loadLabelSet(filename):
 
	binfile = open(filename, 'rb')
	buffers = binfile.read()
 
	head = struct.unpack_from('>II' , buffers ,0)
	imgNum=head[1]
 
	offset = struct.calcsize('>II')
	numString = '>'+str(imgNum)+"B"
	labels = struct.unpack_from(numString , buffers , offset)
	binfile.close()
	labels = np.reshape(labels,[imgNum,1])
	enc = preprocessing.OneHotEncoder()  
	enc.fit(labels)  
	labels=enc.transform(labels).toarray()
	return labels
 

