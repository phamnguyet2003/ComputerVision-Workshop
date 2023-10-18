import numpy as np
import cv2
import time
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
import csv
try:
    import seaborn as sns
    import pandas as pd
except:
    raise Exception('Seaborn or Pandas packages not found. Installation: $ pip install seaborn pandas')

def getInput():
    '''
        Press k to cap and save image in folder name "input_data",
        Press q to quit.

    '''
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture((input("Enter IP Camera (vd: http://192.168.137.211:4747/video)")))
    count = 0
    while(True):
        # Capture frame-by-frame
        start_time = time.time()
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (200, 200))
        # frame = frame[0:640,0:480]
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('k to cap, q to quit',frame)
        cv2.imshow('Press q to quit',frame)
        # Display the resulting frame
        # if cv2.waitKey(1) & 0xFF == ord('k'):
        if cv2.waitKey(200) :
            cv2.imwrite('./raw_data/'+str(count)+'.jpg',frame)
            count += 1

        if (cv2.waitKey(1) & 0xFF == ord('q')) or count == 20:
            break
        # time.sleep(1.0 - time.time() + start_time) # Sleep for 1 second minus elapsed time

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def delete_files_in_directory(directory_path):
    '''
        Function to delete all files in specific directory
        Input: firectory path
    '''
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"File '{file_path}' deleted successfully.")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")

def finish_input():
    '''
        Input the name (name of this person)
        Save the image in directory name which has all preprocessed images
        Path: from 'raw_data' to 'input_data'
    '''
    # known_face_ids = []
    count = 0
    while (True):
        try:
            # get information of student and then update file infor_student
            id = input('Enter the number of student: ')
            name = input("Enter the name: ")
            dob = input('Enter date of birth: ')

            file_path = './infor_student.csv'
            # Open the file for writing using a CSV writer in append mode
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                # Write a row of data to the file
                row = [id,name,dob]
                writer.writerow(row)


            print('Start to preprocess..')
            path = "./input_data/"
            os.mkdir(path+id)
            for image in os.listdir('raw_data'):
                face_image = face_recognition.load_image_file(f'raw_data/{image}')
                cv2.imwrite(f'./input_data/{id}/{str(count)}.jpg',preprocessing(face_image))
                count += 1
                # image: filename
            print('Finish processing and save in directory',name,'has number',id)
            break
        except:
            print("Exist folder!!! Try again!!!")
            continue

def equalize_histogram(img_o):
    '''
        Perform pre-processing steps such as light balance, noise filtering
        Input is an image
        Output is an image which is equalized-histogram
    '''
    def equal_hist(hist):
        cumulator = np.zeros_like(hist, np.float64)
        for i in range(len(cumulator)):
            cumulator[i] = hist[:i].sum()
        #print(cumulator)
        new_hist = (cumulator - cumulator.min())/(cumulator.max() - cumulator.min()) * 255
        #new_hist = np.uint8(new_hist)
        return new_hist

    def compute_hist(img):
        hist = np.zeros((256,), np.uint8)
        h, w = img.shape[:2]
        for i in range(h):
            for j in range(w):
                hist[img[i][j]] += 1
        return hist

    # img_path =  './faces/2.jpg'
    # img_o = cv2.imread(img_path)
    img_he = img_o.copy()
    img_he = cv2.cvtColor(img_he, cv2.COLOR_RGB2HSV)
    origin_v= img_he[:,:,2].reshape(1,-1)[0].copy()

    hist = compute_hist(img_he[:,:,2]).ravel()
    new_hist = equal_hist(hist)

    h, w = img_he.shape[:2]
    for i in range(h):
        for j in range(w):
            img_he[i,j,2] = new_hist[img_he[i,j,2]]

    after_v = img_he[:,:,2].reshape(1,-1)[0].copy()
    img_he = cv2.cvtColor(img_he, cv2.COLOR_HSV2RGB)

    return img_he

def noise_filtering(img):
    '''
        Using median convolution to filter noises.
        Input is an image you want to process
        Return the image which is result after noise filtering.
    '''
    filtered_img = cv2.medianBlur(img, ksize=5)  # Adjust the kernel size as needed
    return filtered_img

def preprocessing (img):
    '''
        Preprocessing: light balance and noise filtering
        Input is an image you want to process
        Return the image which is result after noise filtering.
    '''
    result = equalize_histogram(img)
    result = noise_filtering(result)
    return result

def faceRecognition(train_path, test_path):
    '''
        Function which has inputs is train_path and test_path.
        Return the label of face which was recognized.
    '''
    training_data = []  # 2D array of flattened image arrays
    labels = os.listdir(train_path)  # List of name of labels
    labels_number = []  # 1D array or list of corresponding labels (numeric)

    count = 0
    for classes in os.listdir(train_path):
        for image in os.listdir(train_path + classes):
            image = cv2.imread(train_path + classes+'/'+image)
            training_data.append(image.flatten())
            labels_number.append(count)
        count += 1
    # Create an instance of EigenFaceRecognizer
    recognizer = cv2.face.EigenFaceRecognizer_create()

    # Add your training data and labels to the lists
    recognizer.train(training_data, np.array(labels_number))

    # Perform face recognition on a test image
    test_image = cv2.imread(test_path)
    test_image = cv2.resize(test_image, (640, 480))  # Resize to match the size of training images
    # gray_test_image = cv2.cvtColor(resized_test_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # test_image_flattened = gray_test_image.flatten()
    test_image_flattened = test_image.flatten()

    label, confidence = recognizer.predict(test_image_flattened)

    # Print the predicted label and confidence score
    # print("Predicted Label:", labels[label])
    # print("Confidence:", confidence)
    return labels[label]

def faceRecognition_video(train_path, test_image):
    '''
        Function which has inputs is train_path and test_path.
        Return the label of face which was recognized.
    '''
    training_data = []  # 2D array of flattened image arrays
    labels = os.listdir(train_path)  # List of name of labels
    labels_number = []  # 1D array or list of corresponding labels (numeric)

    count = 0
    for classes in os.listdir(train_path):
        for image in os.listdir(train_path + classes):
            image = cv2.imread(train_path + classes+'/'+image)
            training_data.append(image.flatten())
            labels_number.append(count)
        count += 1
    # Create an instance of EigenFaceRecognizer
    recognizer = cv2.face.EigenFaceRecognizer_create()

    # Add your training data and labels to the lists
    recognizer.train(training_data, np.array(labels_number))

    # Perform face recognition on a test image
    # test_image = cv2.imread(test_path)
    # test_image = cv2.resize(test_image, (640, 480))  # Resize to match the size of training images
    # gray_test_image = cv2.cvtColor(resized_test_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # test_image_flattened = gray_test_image.flatten()
    test_image_flattened = test_image.flatten()
    test_image = cv2.resize(test_image, (640, 480))  # Resize to match the size of training images
    label, confidence = recognizer.predict(test_image_flattened)

    # Print the predicted label and confidence score
    # print("Predicted Label:", labels[label])
    # print("Confidence:", confidence)
    return labels[label]

def takePicture2Test():
    '''
        Press k to cap and save image in folder name "input_data",
        Press q to quit.

    '''
    cap = cv2.VideoCapture(0)
    count = 0
    while(True):
        # Capture frame-by-frame
        start_time = time.time()
        ret, frame = cap.read()
        name = input('Name of picture:')
        # frame = cv2.resize(frame, (200, 200))
        # frame = frame[0:640,0:480]
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('k to cap, q to quit',frame)
        # cv2.imshow('Press q to quit',frame)
        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('k'):
        # if cv2.waitKey(200) :
            cv2.imwrite('./unknown_faces/'+name+'.jpg',frame)
            count += 1

        if (cv2.waitKey(1) & 0xFF == ord('q')) or count == 5:
            break
        # time.sleep(1.0 - time.time() + start_time) # Sleep for 1 second minus elapsed time

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()