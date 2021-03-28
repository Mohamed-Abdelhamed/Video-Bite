import cv2


class DetectEmotionController:
    def __init__(self,smallVideos):
        self.smallVideos = smallVideos
        self.detectFaces = DetectFaces()
        self.emotionModel = EmotionModel()

    #### main function #####
    def extract_emotions(self,secondJump = 1):
        results = []
        for video in self.smallVideos:
           frames = ExtractFrames.extract_frames(video,secondJump)
           faceImages = self.detectFaces.detect(frames)
           results.append(self.emotionModel.predict(faceImages))
        return results
    


class ExtractFrames:
        @staticmethod
        def extract_frames(video,secondJump):
            framesList = []
            cap=cv2.VideoCapture(video)
            if not cap.isOpened():
                print('cap was not opened')
                cap.open(video)
            # count the number of frames. 
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
            # number of frames per second.
            fps = int(cap.get(cv2.CAP_PROP_FPS)) 
            seekFrame=0
            print('framesss::')
            print(frames)
            print('fps::')
            print(fps)
            while seekFrame < (frames-fps):
                    print('in frame oneeeeeeeee')
                    print(seekFrame)
                    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
                    if ret:
                        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
                        framesList.append(gray_img)
                    jump =seekFrame+fps*secondJump
                    cap.set(cv2.CAP_PROP_POS_FRAMES,jump)
                    seekFrame=jump
            return framesList



import dlib
import cv2
class DetectFaces:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
    
    
    def detect(self,frames):
        faceFrames =[]
        for frame in frames:
            faces_detected = self.detector(frame, 1)
            print("Number of faces detected: {}".format(len(faces_detected)))
            for face in faces_detected:
                roi_gray=frame[max(0, face.top()): min(face.bottom(), frame.shape[0]),
                        max(0, face.left()): min(face.right(), frame.shape[1])]#cropping region of interest i.e. face area from  image
                faceFrames.append(roi_gray)
        return faceFrames

import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image


class EmotionModel:
    def __init__(self):
                #load model
        self.model = model_from_json(open("models/Emotion-detection/fer.json", "r").read())
        #load weights
        self.model.load_weights('models/Emotion-detection/fer.h5')


    def predict(self,facesImages):
        arr = [0] * 7
        for frameImage in facesImages:
            roi_gray=cv2.resize(frameImage,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255
            predictions = self.model.predict(img_pixels)

            #find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            print('predicted emotion is')
            print(predicted_emotion)
            arr[max_index]= arr[max_index]+1

        return arr