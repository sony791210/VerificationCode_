# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/python
# -*- coding: utf-8 -*
import tensorflow as tf
import string
import math
from PIL import Image
import numpy as np
import pandas as pd
import random  as rd
import sys,os  
from PIL import Image,ImageDraw 
import outpic  as op



#define the picture size 
#char_num is the number in pic
#the classes is include number(0~9)+captile(A~Z)
class captchaModel():
    def __init__(self,
                 width = 25,
                 height = 24,
                 char_num = 1,
                 classes = 36,
                 characters = string.digits + string.ascii_uppercase):
        self.width = width
        self.height = height
        self.char_num = char_num
        self.classes = classes
        self.characters = characters
        

        
#batch_size i don't know sorry
#the total number is 2500

    def gen_captcha(self,batch_size =50):
        X = np.zeros([batch_size,self.height,self.width,1])
        img = np.zeros((self.height,self.width),dtype=np.uint8)
        Y = np.zeros([batch_size,self.char_num,self.classes])
        data=pd.read_excel('train/data.xlsx')
        while True:
            for i in range(batch_size):
                nn=rd.randint(0,2499)
                captcha_str = str(data['name'][nn])
                img = Image.open('train/'+str(nn)+'.jpg').convert('L')
                img = np.array(img.getdata())
                X[i] = np.reshape(img,[self.height,self.width,1])/255.0
                for j,ch in enumerate(str(captcha_str)):
                    Y[i,j,self.characters.find(ch)] = 1
            Y = np.reshape(Y,(batch_size,self.char_num*self.classes))
            yield X,Y
            


#read CNN u will know 
    def get_parameter(self):
        return self.width,self.height,self.char_num,self.characters,self.classes
    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    def create_model(self,x_images,keep_prob):
        #first layer
        w_conv1 = self.weight_variable([5,5, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.leaky_relu(tf.nn.bias_add(self.conv2d(x_images, w_conv1), b_conv1))
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_dropout1 = tf.nn.dropout(h_pool1,keep_prob)
        conv_width = math.ceil(self.width/2)
        conv_height = math.ceil(self.height/2)
        #second layer
        w_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.leaky_relu(tf.nn.bias_add(self.conv2d(h_dropout1, w_conv2), b_conv2))
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_dropout2 = tf.nn.dropout(h_pool2,keep_prob)
        conv_width = math.ceil(conv_width/2)
        conv_height = math.ceil(conv_height/2)
        #third layer
        w_conv3 = self.weight_variable([5, 5, 64, 64])
        b_conv3 = self.bias_variable([64])
        h_conv3 = tf.nn.leaky_relu(tf.nn.bias_add(self.conv2d(h_dropout2, w_conv3), b_conv3))
        h_pool3 = self.max_pool_2x2(h_conv3)
        h_dropout3 = tf.nn.dropout(h_pool3,keep_prob)
        conv_width = math.ceil(conv_width/2)
        conv_height = math.ceil(conv_height/2)
        #first fully layer
        conv_width = int(conv_width)
        conv_height = int(conv_height)
        w_fc1 = self.weight_variable([64*conv_width*conv_height,1024])
        b_fc1 = self.bias_variable([1024])
        h_dropout3_flat = tf.reshape(h_dropout3,[-1,64*conv_width*conv_height])
        h_fc1 = tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(h_dropout3_flat, w_fc1), b_fc1))
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        #second fully layer
        w_fc2 = self.weight_variable([1024,self.char_num*self.classes])
        b_fc2 = self.bias_variable([self.char_num*self.classes])
        y_conv = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)
        return y_conv