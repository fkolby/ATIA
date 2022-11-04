import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

cwd = os.getcwd()


for folder in ['NORMAL','PNEUMONIA']:
    fl = os.listdir(cwd+'/train/'+folder)
    Xtrain,Xval = train_test_split(fl,test_size=1/6,random_state=442)
    for el in Xval:
        shutil.move(cwd+'/train/'+folder + '/' + el, cwd+'/val/'+folder)