import cv2
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()



X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ## Declare the model
# model = Sequential()

# ## Declare the layers
# layer_1 = Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1))
# layer_2 = Conv2D(64, kernel_size=3, activation='relu')
# layer_3 = Flatten()
# layer_4 = Dense(10, activation='softmax')

# ## Add the layers to the model
# model.add(layer_1)
# model.add(layer_2)
# model.add(layer_3)
# model.add(layer_4)

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)


# model.save('digitModel')


model = load_model('digitModel')

# example = X_train[1]
# prediction = model.predict(example.reshape(1, 28, 28, 1))

# print ("Prediction (Softmax) from the neural network:\n\n {}".format(prediction))

# hard_maxed_prediction = np.zeros(prediction.shape)
# hard_maxed_prediction[0][np.argmax(prediction)] = 1
# print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))

# print ("\n\n--------- Prediction --------- \n\n")
# plt.imshow(example.reshape(28, 28), cmap="gray")
# plt.show()
# print("\n\nFinal Output: {}".format(np.argmax(prediction)))




def ins(digit, arr):
    inserted = False
    for i in range (len(arr)):
        if digit[1] <= arr[i][1]:
            arr.insert(i, digit)
            inserted = True
            break
    

    if(inserted == False):
        arr.insert(len(arr), digit)
    return arr



def readNumber(image):
    image = cv2.imread('img.png')
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preprocessed_digits = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        
        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        
        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]
        
        # Resizing that digit to (18, 18)
        if w < 50:
            print(w)
            resized_digit = cv2.resize(digit,(w//8,18))
            padded_digit = np.pad(resized_digit, ((5,5),(((28-w//8)//2),28-w//8-((28-w//8)//2))), "constant", constant_values=0)
        else:
            print(w)
            resized_digit = cv2.resize(digit, (18,18))
            padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
        
        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        #padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
        
        # Adding the preprocessed digit to the list of preprocessed digits
        ins((padded_digit,x), preprocessed_digits)
    print("\n\n\n----------------Contoured Image--------------------")
    # plt.imshow(image, cmap="gray")
    # plt.show()
        
    #inp = np.array(preprocessed_digits)

    result = ""
    for digit in preprocessed_digits:
        prediction = model.predict(digit[0].reshape(1, 28, 28, 1)) 
        
        result+=str(np.argmax(prediction))

        
        # print ("\n\n---------------------------------------\n\n")
        # print ("=========PREDICTION============ \n\n")
        # plt.imshow(digit[0].reshape(28, 28), cmap="gray")
        # plt.show()
        # print("\n\nFinal Output: {}".format(np.argmax(prediction)))
        
        # print ("\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction))
        
        # hard_maxed_prediction = np.zeros(prediction.shape)
        # hard_maxed_prediction[0][np.argmax(prediction)] = 1
        # print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
        # print ("\n\n---------------------------------------\n\n")
    return result



print(readNumber('he'))