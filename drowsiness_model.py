import sklearn
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
x_train=np.load(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data\x_train.npy")
y_train=np.load(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data\y_train.npy")
x_test=np.load(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data\x_test.npy")
y_test=np.load(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data\y_test.npy")
x_mrl=np.load(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data2\X_mrl.npy")
y_mrl=np.load(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data2\Y_mrl.npy")
index=np.random.choice(x_mrl.shape[0],14000,replace=False)
index2=np.random.choice(x_mrl.shape[0],218,replace=False)
x_mrl1=x_mrl[index]
y_mrl1=y_mrl[index]

x_train=np.concatenate([x_train,x_mrl1])
y_train=np.concatenate([y_train,y_mrl1])
x_mrl2=x_mrl[index2]
y_mrl2=y_mrl[index2]
x_test=np.concatenate([x_test,x_mrl2])/255
y_test=np.concatenate([y_test,y_mrl2])
print(len(x_train))

print(x_train.shape,y_train.shape)


from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
scaler.fit(x_train.reshape(-1,64*64*3))
x_train=scaler.transform(x_train.reshape(-1,64*64*3)).reshape(-1,64,64,3)
#x_test=scaler.transform(x_test.reshape(-1,64*64*3)).reshape(-1,64,64,3)




model=keras.Sequential([keras.layers.Conv2D(filters=32,kernel_size=(4,4),input_shape=(64,64,3),activation="relu"),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(filters=64,kernel_size=(4,4),activation="relu"),
    #keras.layers.MaxPooling2D((2,2)),keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation="relu"),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),keras.layers.Dropout(0.6),
    keras.layers.Dense(2000,activation="relu"),keras.layers.Dropout(0.6),
    keras.layers.Dense(800,activation="relu"),keras.layers.Dropout(0.6),
    keras.layers.Dense(100,activation="relu"),keras.layers.Dropout(0.6),
    keras.layers.Dense(1,activation="sigmoid")])

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=60,batch_size=500)
print(model.evaluate(x_test,y_test))
y_pred=np.round(model.predict(x_test))
#print(y_pred)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

score=accuracy_score(y_test,y_pred)
matrix=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)

print(score)
print(matrix)
print(report)


model.save(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\drowsiness_model_final")