import numpy as np
import matplotlib.pyplot as plt 

data = np.load('./train_and_test.npz')
# print(data.files)
# https://www.kaggle.com/kymo9890/image-class
# https://github.com/galenballew/Traffic-Sign-CNN/blob/master/Traffic_Sign_Classifier.ipynb
X_train, y_train, X_test = data['X_train'], data['y_train'], data['X_test']
# print(X_train.shape[0])
# print(y_train.shape[0])


X = X_train
print(X)
print(X.shape)
X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]


# for i in range(1, 39209, 1000):
#     plt.imshow(X_train[i]) 
#     plt.show()
# https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/
# https://navoshta.com/traffic-signs-classification/
# https://github.com/mohamedameen93/German-Traffic-Sign-Classification-Using-TensorFlow/blob/master/Traffic_Sign_Classifier.ipynb
# # detecting circles with opencv 

# # modified from https://www.geeksforgeeks.org/circle-detection-using-opencv-python/

# import cv2 

# def blur(img):
#     return cv2.blur(img, (3,3))


# def gray(img):
#     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# def circles(img, draw_on_image=False):
#     gray_blurred = blur(gray(img))
    
#     detected_circles = None
    
#     min_thresh = 20
#     while min_thresh > 1 and detected_circles is None:
#         min_thresh -= 1
#         detected_circles = cv2.HoughCircles(gray_blurred, 
#                                             method=cv2.HOUGH_GRADIENT, 
#                                             dp=1, 
#                                             minDist=20, 
#                                             param1=20,
#                                             param2=min_thresh,
#                                             minRadius=7,
#                                             maxRadius=30) 
#     # Convert the circle parameters a, b and r to integers. 
#     detected_circles = np.uint16(np.around(detected_circles)) 
#     if draw_on_image:
#         out = img.copy()
#         for pt in detected_circles[0, :]: 
#             a, b, r = pt[0], pt[1], pt[2] 

#             # Draw the circumference of the circle. 
#             cv2.circle(out, (a, b), r, (0, 255, 0), 2) 

#             # Draw a small circle (of radius 1) to show the center. 
#             cv2.circle(out, (a, b), 1, (0, 0, 255), 3) 
#         return out
#     else:
#         return detected_circles[0][0]

    
# # https://stackoverflow.com/a/47629363/2821370
# def background_subtract_circle(img, circle):
#     x,y,r = circle
#     mask = np.zeros_like(img[:,:,:])
#     cv2.circle(mask, (x,y), r, (255,255,255), -1, 8, 0)
#     out = img & mask
#     return out

# def crop_image(img):
#     return background_subtract_circle(img, circles(img))

# # point = X_train[700]

# #plt.imshow(circles(point, True), cmap='gray', vmin=0, vmax=255)
# #plt.imshow(edges(point), cmap='gray', vmin=0, vmax=255)

# X_train_cropped = [gray(crop_image(img)) for img in X_train]
# plt.imshow(X_train_cropped[700], cmap='gray', vmin=0, vmax=255);

# plt.figure(figsize=(20, 5))
# vals = len(np.unique(y_train))
# plt.hist(y_train, bins=range(0,vals))
# plt.xticks(range(0, vals));

# # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# # https://keras.io/getting-started/functional-api-guide/
# from keras.layers import Input, Dense
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense
# from keras.models import Model

# input_img = Input(shape=(32, 32, 1))

# output_1 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(input_img)
# output_1_pool = MaxPooling2D((2,2))(output_1)

# output_2 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(output_1_pool)
# output_pool = MaxPooling2D((3,3))(output_2)

# img_output = Flatten()(output_pool)

# digits_1 = Dense(64, activation='relu')(img_output)
# digits_2 = Dropout(0.1)(digits_1)
# digits_3 = Dense(64, activation='relu')(digits_2)

# digits_output = Dense(43, activation='softmax')(digits_3)

# model = Model(inputs=input_img, outputs=digits_output)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.summary()

# # https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
# from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical
# from keras.preprocessing.image import ImageDataGenerator


# xtrain, xtest, ytrain, ytest = train_test_split(X_train_cropped, y_train, test_size=0.2)
# ytrain = to_categorical(ytrain)
# ytest = to_categorical(ytest)

# xtrain = np.array(xtrain)
# xtest = np.array(xtest)
# ytrain = np.array(ytrain)
# ytest = np.array(ytest)

# xtrain = xtrain.reshape(xtrain.shape[0], 32, 32, 1)
# xtest = xtest.reshape(xtest.shape[0], 32, 32, 1)

# batch_size = 16

# # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# # this is the augmentation configuration we will use for training
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=False)

# # this is the augmentation configuration we will use for testing:
# # only rescaling
# test_datagen = ImageDataGenerator(rescale=1./255)

# model.fit_generator(
#     train_datagen.flow(xtrain, ytrain, batch_size=32), 
#     #steps_per_epoch=len(xtrain) / 32,
#     validation_data=test_datagen.flow(xtest, ytest, batch_size=32),
#     epochs=200)

# # model.fit(xtrain, ytrain,
# #           epochs=200,
# #           batch_size=16,
# #           validation_data=(xtest, ytest))
# # model.save_weights('model.h5')