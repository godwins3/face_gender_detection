import cv2

#create a function faceBox which creates bounding boxes around the face that is detected.
def faceBox(faceNet,frame):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0 , (227,227), [104,117,123], swapRB = False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > 0.7:
            x1 = int(detection[0,0,i,3]*frame_width)
            y1 = int(detection[0,0,i,4]*frame_height)
            x2 = int(detection[0,0,i,5]*frame_width)
            y2 = int(detection[0,0,i,6]*frame_height)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame, bboxs

#load 'opencv_face_detector.pbtxt' and 'opencv_face_detector_uint8.pb'
faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'

#initialize gender protocol buffer and model
genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'

#load the networks
faceNet = cv2.dnn.readNet(faceModel,faceProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)

#create gender_list and also model _mean_value 
# as this is required while creating blobs from an image.
gender_list = ['Male', 'Female']
model_mean_value = (78.4263377603,87.7689143744, 114.895847746)

#Open Webcam and start detecting Gender and Age of person present in frame
video  = cv2.VideoCapture(0)
padding = 20

while True:
    # reads frame from a camera 
    ret,frame= video.read()
    frame,bboxs = faceBox(faceNet,frame)
    for bbox in bboxs:
        face= frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        blob = cv2.dnn.blobFromImage(face,1.0 , (227,227),model_mean_value,swapRB = False)
        
        
        genderNet.setInput(blob)
        gender_pred = genderNet.forward()
        gender = gender_list[gender_pred[0].argmax()]
        
        label = "{}".format(gender)
        cv2.rectangle(frame, (bbox[0], bbox[1]-10), (bbox[2], bbox[1]),(0,255,0),-1)
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2, cv2.LINE_AA)
    
    if ret:
        cv2.imshow('ProjectPraise Face-Gender', frame)
        

    cv2.imshow('ProjectPraise Face-Gender', frame)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
