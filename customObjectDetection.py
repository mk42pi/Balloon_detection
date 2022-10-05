import cv2
import numpy as np

confThreshold = 0.5 # Confidence threshold
nmsThreshold = 0.4 # Non-maximum suppression threshold
inputWidth = 320 # Width of the input 
inputHeight = 320 # Height of the input 

className = []
with open('classes.txt', 'r') as f:
    className = f.read().splitlines()
    

modelConf = 'yolov3_testing.cfg'
modelWeights = 'yolov3_copy_final.weights'

net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

cap = cv2.VideoCapture('video.mp4')

def findObject(output, video):
    heightV, widthV, channelV = video.shape
    
    classIds = []
    confidences = []
    boxes = []
    
    for out in output:
        for det in out:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            
            if confidence > confThreshold:
                
                width = int(det[2] * widthV)
                height = int(det[3] * heightV)                
                center_x = int((det[0] * widthV) - width/2)
                center_y = int((det[1] * heightV) - height/2)

                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([center_x, center_y, width, height])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    for i in indices:
        box = boxes[i]
        center_x, center_y = box[0], box[1]
        width, height = box[2], box[3]
        
        cv2.rectangle(video, (center_x, center_y), (center_x + width, center_y + height), (100, 0, 255),1)
        cv2.putText(video, f'{className[classIds[i]]} {int(confidences[i]*100)}%',
                    (center_x, center_y), cv2.FONT_HERSHEY_PLAIN, 1, (100, 0, 255), 2)


while True:
    
    success, video = cap.read()
    video = cv2.resize(video, None, fx = 0.8, fy = 0.8)
    # Stop the program if reached end of video
    if not success:
        print('Done processing!')
        break
    
    blob = cv2.dnn.blobFromImage(video, 1/255, (inputWidth, inputHeight), (0, 0, 0), 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    
    findObject(outputs, video)
    
    cv2.imshow('Find Balloon', video)
    
    if cv2.waitKey(16) and 0xFF == ord('q'):
        break
    
    