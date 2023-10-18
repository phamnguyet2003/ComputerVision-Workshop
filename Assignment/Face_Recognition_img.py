from lib import *
import face_recognition
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageTk, Image
import cv2
import numpy as np

# train_path = './input_data/'
# test_path = './unknown_faces/34.jpg'
# print(faceRecognition(train_path, test_path))


# im1, im2 are links of 2 images
global im1
im1 = None

def resize(image, width):
    # Define the new size (width, height) for the image
    new_size = (width, int(image.shape[0] * (width / image.shape[1])))

    # Resize the image
    resized_image = cv2.resize(image, new_size)
    
    return resized_image



# Get information of the student in csv file

def getInfomation (id):
    '''
        This function will get input is id of person, 
        return name and date of birth of this person.
    '''
    # global data_student
    path_infor = "infor_student.csv"
    data_student = pd.read_csv(path_infor)
    data_student = pd.DataFrame(data_student)
    if id == "":
        return 'Unknown', 'Unknown'
    if id[-4:] == '.jpg':
        id = id[:-4]
    # id = int(id)
    # id = ' '.join(id.split('.')[:-1])
    name = data_student['Name'][data_student['Number'] == id].to_string(index=False)
    dob = data_student['Date of Birth'][data_student['Number'] == id].to_string(index=False)
    return name, dob

def on_button_clicked():
    global im1
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename()
    if filename:
        im1 = filename
        print('You choose:', filename)

def on_button_clicked2():
    global im1 # path of image
    train_path = './input_data/'
    test_path = im1
    id = (faceRecognition(train_path, test_path))
    name, dob = getInfomation(id)

    print('Name of student: ',name)
    print('Number of student: ',id)
    print('Date of birth of student: ',dob)
    img = cv2.imread(im1)
    # Converting image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Loading the required haar-cascade xml classifier file
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Applying the face detection method on the grayscale image
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)

    # Iterating through rectangles of detected faces
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, id, (x,y+h+22), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)


    cv2.imshow('Detected faces', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def on_button_clicked3():
    directory_path = './raw_data'
    delete_files_in_directory(directory_path)
    getInput()
    finish_input()
    directory_path = './raw_data'
    delete_files_in_directory(directory_path)

def on_button_clicked4():
    takePicture2Test()

root = tk.Tk()
root.geometry('600x600')

# Create a canvas
canvas = tk.Canvas(root, width=600, height=300)
canvas.pack(fill='both', expand=True)

# Add a button to the canvas
browse_button = tk.Button(canvas, text='Browse Picture', command=on_button_clicked, width=15, height = 1)
browse_button.pack(pady=200)
browse_button.place(x=350, y=535)

# Add a button to the canvas
browse_button2 = tk.Button(canvas, text='Face Recognition', command=on_button_clicked2, width=15, height = 1)
browse_button2.pack(pady=200)
browse_button2.place(x=475, y=535)

# Add a button to the canvas
browse_button3 = tk.Button(canvas, text='Take input', command=on_button_clicked3, width=15, height = 1)
browse_button3.pack(pady=200)
browse_button3.place(x=225, y=535)

# Add a button to the canvas
browse_button4 = tk.Button(canvas, text='Take picture to test', command=on_button_clicked4, width=15, height = 1)
browse_button4.pack(pady=200)
browse_button4.place(x=350, y=565)

# Load the background image
# bg_image = Image.open(r'D:\Nguyet Folder\CPV301\Lab5\eiffel.jpeg')
bg_image = Image.open('background.webp')
bg_image = bg_image.resize((600, 500), Image.ANTIALIAS)
bg_image_tk = ImageTk.PhotoImage(bg_image)

# Add the background image to the canvas
canvas.create_image(0, 0, image=bg_image_tk, anchor='nw')
# Add some text to the canvas
text = 'Face Recognition\n Program'
canvas.create_text(100, 550, text=text, font=('Arial', 17), fill='black')

def on_closing():
    root.destroy()
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()