# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:47:54 2018

@author: apple
"""

#!/usr/bin/python
from PIL import Image, ImageFilter
import tensorflow as tf
import numpy as np
import string
import sys
import captcha_model
import pandas as pd
import outpic  as op

    
    
    
def get_data(gray_image):
    tf.reset_default_graph()
    captcha = captcha_model.captchaModel()
    width,height,char_num,characters,classes = captcha.get_parameter()
    img = np.array(gray_image.getdata())
    test_x = np.reshape(img,[height,width,1])/255.0
    x = tf.placeholder(tf.float32, [None, height,width,1])
    keep_prob = tf.placeholder(tf.float32)
    model = captcha_model.captchaModel(width,height,char_num,classes)
    y_conv = model.create_model(x,keep_prob)
    predict = tf.argmax(tf.reshape(y_conv, [-1,char_num, classes]),2)
    
    
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True


    
    with tf.Session(config=config) as sess:
        
        sess.run(init_op)   
        saver.restore(sess, "./capcha_model.ckpt")
        pre_list =  sess.run(predict,feed_dict={x: [test_x], keep_prob: 1})
        for i in pre_list:
            s = ''
            for j in i:
                s += characters[j]
    return s


if __name__ == '__main__':
    
    
    ss=pd.Series()
    for j in range(1):
        gray_image = Image.open('test/'+str(j+1)+'.jpg').convert('L')
        op.clearNoise(gray_image,30,3,7)
    #gray_image.show()
        w = ''
        for i in range(5):
            box=0+25*i,0,25+25*i,24
            image=gray_image.crop(box)
            op.clearNoise(image,60,3,7)
            #image.show(
            s=get_data(image)
            w=w+str(s)
        ss=ss.append(pd.Series(w)).reset_index(drop=True)
        print(ss)
    
    
    
    
    