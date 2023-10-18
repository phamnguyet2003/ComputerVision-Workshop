import face_recognition
import cv2
import numpy as np
import math
import os, sys
import pandas as pd
import datetime
import csv

''' 
    Read csv file into a dataframe name data_student
    File infor_student.csv has information of all people you want to check attendence
'''
global data_student
data_student = pd.read_csv("infor_student.csv")
data_student = pd.DataFrame(data_student)

# Get information of the student in csv file

def getInfomation (id):
    '''
        This function will get input is id of person, 
        return name and date of birth of this person.
    '''
    if id == "":
        return 'Unknown', 'Unknown'
    if id[-4:] == '.jpg':
        id = id[:-4]
    id = int(id)
    # id = ' '.join(id.split('.')[:-1])
    name = data_student['Name'][data_student['Number'] == id].to_string(index=False)
    dob = data_student['Date of Birth'][data_student['Number'] == id].to_string(index=False)
    return name, dob

def face_confidence(face_distance, face_match_threshold = 0.8):
    '''
        This function will calculate the percent of same ratio between face in webcam
        and face which has the face_distance is min.
    '''
    range = 1.0 - face_match_threshold
    linear_val = (1.0 - face_distance)/(range*2.0)

    if face_distance > face_match_threshold:
        return round(linear_val * 100, 2)
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5)*2, 0.2))) * 100
        return round(value,2)

def preprocessing (img):
    '''
        This function will preprocess the image using white balance and noise filtering
        Return the preprocessed image
    '''
    # Perform white balance
    result_wb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform noise filtering
    result_nf = cv2.medianBlur(result_wb, 5)
    return result_nf

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_ids = []
    known_face_ids = []
    known_face_encodings = []
    process_current_frame = True
    attendence = []
    faces_detected = []
    attendence_images = []

    def __init__(self):
        self.encode_faces()

    def encode_faces (self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encodings = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encodings)
            self.known_face_ids.append(image)

        # print(self.known_face_ids)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)
        isFirst = True
        while True:
            # global data_student
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                # find all faces in the current frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_ids = []

                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    id = '######'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        id = self.known_face_ids[best_match_index]
                        # confidence = face_confidence(face_distances[best_match_index])
                    
                    # self.face_ids.append(f'{id}({confidence})')
                    self.face_ids.append(f'{id}')
                    # name = data_student['Name'][data_student['Number'] == id].to_string(index=False)
                    name = 'Unknown'
                    dob = 'Unknown'
                    if id != '######':   
                        name = getInfomation(id)[0]
                        dob = getInfomation(id)[1]
                        if not any(face_detected['id'] == id for face_detected in self.faces_detected):
                            current_time = datetime.datetime.now()
                            current_date = current_time.date()
                            current_time = current_time.time().strftime("%H:%M:%S")
                            self.faces_detected.append({'id': id, 'isDetected': True})
                            cv2.imwrite(f'./attendence/attendence_img/{name}.jpg', preprocessing(rgb_small_frame))
                            self.attendence_images.append(preprocessing(rgb_small_frame))
                            # write file attendence.csv
                            file_path = './attendence/attendence_check_csv/attendence.csv'
                            # Open the file for writing using a CSV writer in append mode
                            with open(file_path, 'a', newline='') as f:
                                writer = csv.writer(f)
                                # Write a row of data to the file
                                row = [id,name,dob,current_date,current_time]
                                writer.writerow(row)
                        
                    if name not in self.attendence :
                        self.attendence.append(name)
                        current_time = datetime.datetime.now()
                        current_date = current_time.date()
                        current_time = current_time.time().strftime("%H:%M:%S")

                        print(f'{name} - {dob} - {current_date} - {current_time}')

            self.process_current_frame = not self.process_current_frame

            # display annotations
            for (top, right, bottom, left), id in zip(self.face_locations, self.face_ids):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                id = ' '.join(id.split('.')[:-1])
                name = 'Unknown'
                dob = 'Unknown'
                if id != '######':   
                    name = getInfomation(id)[0]
                    dob = getInfomation(id)[1]
                # dob  = getInfomation(id)
                cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), -1)
                cv2.putText(frame, str(name), (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)


            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
