
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import keras
import tensorflow as tf


def append_n_images(list,path,n,img_class):
    for i in range(n):

        try:
            file=path+str(i)+".jpg"
            img=cv2.imread(file)
            #print(img.shape)
            img=cv2.resize(img,(64,64))
            #print(img.shape)
            list.append((img,img_class))
            '''
            if (i==3 and (path==folder_test_closed or path==folder_test_open)):
                print(type(img))
                plt.imshow(img)
                plt.show()
            elif (i==1 and(path==folder_closed or path==folder_open)):
                print(type(img))
                plt.imshow(img)
                plt.show()
            '''

        except:
            pass
    return list


def get_x_and_y(list):
    listx=[]
    listy=[]
    for data,img_class in list:
        listx.append(data)
        listy.append(img_class)

        
    arr_x=np.array(listx)
    arr_y=np.array(listy)

    return arr_x,arr_y


folder_closed="C:\\Users\\Gurmehar\\Desktop\\Data Science\\ML\ML3\\dataset_new\\train\\Closed\\_"
folder_open="C:\\Users\\Gurmehar\\Desktop\\Data Science\\ML\\ML3\\dataset_new\\train\\Open\\_"

folder_test_closed="C:\\Users\\Gurmehar\\Desktop\\Data Science\\ML\\ML3\\dataset_new\\test\\Closed\\_"
folder_test_open="C:\\Users\\Gurmehar\\Desktop\\Data Science\\ML\\ML3\\dataset_new\\test\\Open\\_"


list_closed=append_n_images([],folder_closed,726,1)
list_training=append_n_images(list_closed,folder_open,726,0)

list_test_closed=append_n_images([],folder_test_closed,720,1)
list_testing=append_n_images(list_test_closed,folder_test_open,720,0)

print(len(list_training),len(list_testing))


x_train,y_train=get_x_and_y(list_training)
x_test,y_test=get_x_and_y(list_testing)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

np.save(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data\x_train",x_train)
np.save(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data\x_test",x_test)
np.save(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data\y_train",y_train)
np.save(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data\y_test",y_test)

X=np.concatenate([x_train,x_test])
Y=np.concatenate([y_train,y_test])
np.save(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data\X",X)
np.save(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data\Y",Y)
print(X.shape)
print(Y.shape)

#1234 218