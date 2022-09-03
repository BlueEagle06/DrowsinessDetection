import pandas as pd
import numpy as np
import os
import cv2
images_list=[]
label_list=[]
value_error=ValueError()
def images(folder):
    images_list=[]
    label_list=[]
    for filename in os.listdir(folder):
        list_filename=filename.split("_")
        path=os.path.join(folder,filename)
        img=cv2.imread(path)
        #print(img.shape)
        img=cv2.resize(img,(64,64))
        #print(img.shape)

        images_list.append(img)
        if list_filename[4] =="0":
            label_list.append(1)
        elif list_filename[4]=="1":
            label_list.append(0)

        else:
            raise value_error("dd")

        #images_list.append()

        #path=os.path.join(folder,filename)
        #mages_list.append(cv2.imread(path))
    
    return(np.array(images_list),np.array(label_list))



images_arr,labels_arr=images(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data\mrlEyes_2018_01\s0001")
print(images_arr.shape,labels_arr.shape)




for i in range(2,38):
    print(i)
    if i<10:
        folder="C:\\Users\\Gurmehar\\Desktop\\Data Science\\ML\\ML3\\data\\mrlEyes_2018_01\\s000"+str(i)
    else:
        folder="C:\\Users\\Gurmehar\\Desktop\\Data Science\\ML\\ML3\\data\\mrlEyes_2018_01\\s00" +str(i)

    arr_imgs,arr_lbls=images(folder=folder)
    print(arr_imgs.shape,arr_lbls.shape)
    images_arr=np.concatenate([images_arr,arr_imgs])
    labels_arr=np.concatenate([labels_arr,arr_lbls])

print(images_arr.shape,labels_arr.shape)

np.save(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data2\X_mrl",images_arr)
np.save(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data2\Y_mrl",labels_arr)
    


