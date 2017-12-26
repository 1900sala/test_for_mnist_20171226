
# coding: utf-8

# In[1]:


from spp_net_f import *
from spp_layer_f import *
import os
import time
import input_data
import tensorflow as tf
import numpy as np



train_size = 3060
batch_size = 50
max_epochs =2
num_class = 10
eval_frequency = 100
max_steps = 10000




# In[2]:


class data_label_data():
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def next_batch(self, batch_size):
        label_len = len(self.label)
        shuffle_index = np.arange(label_len)
        np.random.shuffle(shuffle_index)
        batch_index = shuffle_index[:batch_size]
        data_batch = self.data[batch_index]
        label_batch = self.label[batch_index]
        return [data_batch, label_batch]
class load_data(data_label_data):
    def __init__(self, train, test):
        self.train = train
        self.test = test

def train():
    global_step = tf.Variable(0, trainable=False)
    spp_net = SPPnet()
    spp_net.set_lr(0.0001, batch_size, train_size)
    
# load data
    print('load data')
#     train_data = np.load("mnist_train_data.npy")
#     train_label = np.load("mnist_train_label.npy")
#     test_data = np.load("mnist_test_data.npy")
#     test_label = np.load("mnist_test_label.npy")
#     train_part = data_label_data(train_data, train_label)
#     test_part = data_label_data(test_data, train_label)
#     mnist = load_data(train_part, test_part)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
    x_mag = tf.placeholder("float", shape=[None, 784])
    train_data = tf.reshape(x_mag, [-1,28,28,1])
    train_label = tf.placeholder("float", shape=[None, 10])
    keep_prob = tf.placeholder("float")
    num_class = 10
    print("load done")


# train
    print('train')
 
    logits = spp_net.inference(train_data, True, num_class, tp='spp')
    loss, accuracy, opt  = spp_net.train(logits, global_step, train_label)
    print('train done')
    
# evaluation
#    eval_logits = spp_net.inference(valid_data, False, num_class)
#    eval_accuracy = spp_net.loss(eval_logits, valid_label)
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()
        sess.run(init)        
        start_time = time.time()
    #    print((FLAGS.max_epochs * train_size) // batch_size)
        for step in range(max_steps):
            batch = mnist.train.next_batch(batch_size)
            opt.run( feed_dict={ x_mag:batch[0], train_label: batch[1], keep_prob: 0.5 } )
            if step % eval_frequency ==0:
                stop_time = time.time() - start_time
                start_time = time.time()
                loss_value=loss.eval(feed_dict={x_mag:batch[0], train_label: batch[1], keep_prob: 1.0})
                accu = accuracy.eval(feed_dict={x_mag:batch[0], train_label: batch[1], keep_prob: 1.0})
                print('epoch: %.2f , %.2f ms' % (step * batch_size /train_size,
                    1000 * stop_time / eval_frequency)) 
                print('train loss: ', loss_value) 
                print('train accu: ', accu)         


if __name__ == '__main__':
    train()



