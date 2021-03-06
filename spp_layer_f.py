
# coding: utf-8

# In[ ]:


import tensorflow as tf
import math

class SPPLayer():
    def __init__(self,bins,feature_map_size):
        self.h = []
        self.w = []
#        print(type(feature_map_size))
        self.feature_map_size = feature_map_size
        self.bins = bins
        self.n = len(bins)

    def spatial_pyramid_pooling(self,data):
        self.input = data
        self.map_nums=self.input.get_shape().as_list()[3]
        for i in range(self.n):
            x = math.ceil(self.feature_map_size[0]/float(self.bins[i]))
            self.h.append(x)
            x = math.ceil(self.feature_map_size[1]/float(self.bins[i]))
            self.w.append(x)


        self.pooled_out = []
        for i in range(self.n):
            self.pooled_out.append(tf.nn.max_pool(self.input,
                ksize=[1, self.h[i], self.w[i], 1],
                strides=[1, self.h[i], self.w[i], 1],
                padding='SAME'))
        #print('pooled_out', self.pooled_out)
        for i in range(self.n):
            #print(self.bins[i], self.bins[i]*self.bins[i]*self.map_nums)
            #print(self.pooled_out[i].shape)
            self.pooled_out[i] = tf.reshape(self.pooled_out[i], [-1, self.bins[i]*self.bins[i]*self.map_nums])

        #print('pooled_out', self.pooled_out)
        self.output = tf.concat( [self.pooled_out[0], self.pooled_out[1], self.pooled_out[2]],1)
        #print(self.output.shape)
        return self.output

