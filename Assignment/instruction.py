from lib import *
import cv2
import matplotlib.pyplot as plt
import face_recognition
import os

############## How to use preprocessing ###############
# img_path =  './input_data/2.jpg'
# img_o = cv2.imread(img_path)
# img_he = preprocessing(img_o)
# fig, ax = plt.subplots(1, 2, figsize=(15, 10))
# ax[0].imshow(img_o)
# ax[1].imshow(img_he)
# ax[0].set_title("Original Image")
# ax[1].set_title("Histogram Equalization Image")
# plt.show()
# print(img_o - img_he)

############## How to get input ###############
# getInput()

############## How to preprocess all input images ###############
# finish_input()

############## How to delete all files in specific directory ###############
# directory_path = input("Enter path of the directory which has all files you want to delete:")
# directory_path = './raw_data'
# delete_files_in_directory(directory_path)

############## How to use model to recognize the face in test image ###############
train_path = './input_data/'
test_path = './unknown_faces/34.jpg'
print(faceRecognition(train_path, test_path))
