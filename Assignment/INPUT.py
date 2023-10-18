from lib import *
import cv2
import matplotlib.pyplot as plt
import face_recognition
import os

'''
    All function in file lib.py
    After the camera is on, you press K to take picture and press Q when you want to camera off.
    After that, enter the name of person.
'''
directory_path = './raw_data'
delete_files_in_directory(directory_path)
getInput()
finish_input()
directory_path = './raw_data'
delete_files_in_directory(directory_path)