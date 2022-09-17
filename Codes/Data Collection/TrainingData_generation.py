# https://pyimagesearch.com/2014/09/15/python-compare-two-images/

# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

#set the which Nth frames you want to consider
drop_frames = 12 #12
face_classifier = cv2.CascadeClassifier(r'G:\Jupyter Notebook Workspace\GLIM-internship\haarcascade_frontalface_default.xml')
#set the the ssim threshold between -1 to 1 where higher ssim value implies higher similarity in the images
ssim_thresh = 0.92 #set this according to the light conditions #my room need 0.74 threshold #dining room needs 0.9 threshold

subject_id = input("Enter subject id: ")

#CAPTURE FROM WEBCAM
# cap = cv2.VideoCapture(r"G:\Jupyter Notebook Workspace\GLIM-internship\Task 3 _ New subject data generation\captured\working batch\rec_sub7.mp4")
cap = cv2.VideoCapture(0)
# We need to check if camera
# is opened previously or not
if (cap.isOpened() == False): 
    print("Error reading video file")

# cap.set(3, 480)
# cap.set(4, 640)


# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
print(size)
# Below VideoWriter object will createSor
# a frame of above defined The output 
# is stored in 'filename.avi' file.


# result = cv2.VideoWriter(r"Task 3 _ New subject data generation\captured\{}_training_video.avi".format(subject_id),
#                         cv2.VideoWriter_fourcc(*'MJPG'),
#                         12, 
#                         size)

firstFrameCaptured = False
framesCaptured = 0
frameNumber = 0
prev_frame = np.zeros((frame_height, frame_width), dtype = "uint8")

while True:
    ret, new_frame = cap.read()
    # result.write(new_frame)
    if not ret:
        print("no frame captured")
        break
    
    frameNumber += 1
    if firstFrameCaptured == False:
        firstFrameCaptured = True

    gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    # cv2.circle(new_frame, (560,300), radius=5, color=(0, 0, 255), thickness=1)
    frame_rgb = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
    
    for(x,y,w,h) in faces:
        # cv2.rectangle(new_frame, (x,y), (x+w, y+h), (0,255,255), 2)
        # if x>500 and y>300 and x<850:
        roi_rgb = frame_rgb[y:y+h, x:x+w]
        roi_rgb = cv2.resize(roi_rgb, (128,128), interpolation=cv2.INTER_AREA)
        # cv2.line(new_frame, (950,0), (950,720), (0,255,0), 2)
        # cv2.line(new_frame, (500,0), (500,720), (0,255,0), 2)
        
        if np.sum([roi_rgb]) != 0:
            # print(cap.get(3))
            # print(cap.get(4))

            cv2.putText(new_frame, "{}".format(framesCaptured), (2,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            if cv2.waitKey(1) & 0xFF == ord('c'):
                framesCaptured = framesCaptured + 1
                cv2.imwrite(r"G:\Jupyter Notebook Workspace\GLIM-internship\Datasets\dataset2\{}\{}_{}.jpg".format(subject_id, subject_id,frameNumber), new_frame[max(0,y-40):min(720,y+h+30), max(0,x-30):min(1280,x+w+30)]) 
                print("CAPTURED ", framesCaptured, "Manually")
            cv2.rectangle(new_frame, (x-31,y-41), (x+w+30, y+h+30), (0,255,255), 1) 
            # if frameNumber%drop_frames == 0:
            if frameNumber%drop_frames == 0:    
                new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                if ret and firstFrameCaptured:
                    # print(prev_frame.shape)
                    # print(new_frame_gray.shape)
                    s = ssim(prev_frame, new_frame_gray)
                    print("s = ",s)
                    if s <= ssim_thresh:
                        framesCaptured = framesCaptured + 1
                        print("[",x-30,",",y-40,"] [",x+w+30,",",y+h+30,"]")
                        print("w = ", w,"  h = ", h)
                        # cv2.imwrite(r"G:\Jupyter Notebook Workspace\GLIM-internship\Datasets\dataset2\{}\{}_{}.jpg".format(subject_id, subject_id,frameNumber), new_frame[max(0,y-40):min(720,y+h+30), max(0,x-30):min(1280,x+w+30)])  
                        print("CAPTURED ", framesCaptured, "  s = ",s)
                    
                prev_frame = new_frame_gray

    cv2.imshow("Face Recognition", new_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
