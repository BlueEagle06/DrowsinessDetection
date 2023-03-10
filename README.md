# DrowsinessDetection
# order in which to read/execute : data preparation, data2, model, app

#The goal of this project is to build a ML model to help detect drowsy drivers in real-time. The model predicts whether the eyes of the driver are open or closed.
drowsinessapp.py takes the ML model and combines it with haarcascades to create the final app. the ML model takes in real-time pictures of the eyes detected using
haarcascades and makes its predictions. A "score" is used to keep track of the amount of time the eyes have been closed for. If the eyes are predicted as "closed", 
the score increases. If they are predicted as "open", the score decreases. If the score exceeds a certain threshold, an alarm is played to alert the driver.



#The first two files are for data importing and data wrangling.

#drowsiness_model.py contains the code for the CNN trained on multiple public datasets of eye images.

## limitations and Future Scope
The minor issue in performance is due to inaccuracy of the haarcascades in detecting the eyes, and not due to the model itself. Therefore we could use a better eye-detection algorithm to improve performance.
