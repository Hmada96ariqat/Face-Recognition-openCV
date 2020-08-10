import cv2
import os
import numpy as np

subjects= ["", "Hmada" ,"Sholy"]
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #load OpenCV face detector
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1);
    if (len(faces) == 0):
        return None, None  
    #extract the face area
    (x, y, w, h) = faces[0]  
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path): 
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        
        
        if not dir_name.startswith("a"):
            continue;
      
        label = int(dir_name.replace("a", ""))
        
       
        subject_dir_path = data_folder_path + "/" + dir_name
        
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path) 
            cv2.imshow("Training on image...", cv2.resize(image, (500, 600)))
            cv2.waitKey(50)
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img,rect,):
    (x, y, w, h) = rect

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
  
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y ), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img,n):
    
    img = test_img.copy()
    face, rect = detect_face(img) 
    label, confidence = face_recognizer.predict(face)
    label_text = subjects [n]
    draw_rectangle(img ,rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(label)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(confidence)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(label_text)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(face)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
    return img 
print("Predicting images.....")
test_img1 = cv2.imread("test-data/test1.jpeg")
test_img2 = cv2.imread("test-data/test2.jpeg")

predicted_img1 = predict(test_img1,1)
cv2.waitKey(1)
predicted_img2 = predict(test_img2,2)
print("Prediction completed")

cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()

