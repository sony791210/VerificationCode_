# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 17:39:00 2018

@author: apple
"""
import outpic  as op
import sys,os  
from PIL import Image,ImageDraw 
import pandas as pd


data=pd.read_excel('SampleAll/data.xlsx')
ss=pd.Series()
nn=0

#total sample is 500
#all sample in SampleAll
#use op.clearNoise to clean the noise
#i cute the picture equal distribution 5 picture  in box

for i in range(500):
    captcha_str = str(data['name'][i])
    img = Image.open('SampleAll/'+str(i+1)+'.jpg').convert('L')
    op.clearNoise(img,20,3,20)
    #img.show()
    for j in range(5):
        box=0+25*j,0,25+25*j,24
        image=img.crop(box)
        image.save('train/'+str(nn)+'.jpg')
        ss=ss.append(pd.Series(captcha_str[j])).reset_index(drop=True)
        nn=nn+1
      

writer = pd.ExcelWriter('train/data.xlsx')
ss.to_excel(writer,'Sheet1')
writer.save()