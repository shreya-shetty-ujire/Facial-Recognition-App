#kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


# Build app and layout
class CamApp(App):
    def build(self):
        # main layout
        self.webcam=Image(size_hint=(1,.8))
        self.button=Button(text="Verify",on_press=self.verify, size_hint=(1,.1))
        self.verification_label=Label(text="Verification Uninitiated",size_hint=(1,.1))

        # Add items to layout
        layout=BoxLayout(orientation='vertical')
        layout.add_widget(self.webcam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #load keras model
        self.model=tf.keras.models.load_model('siamesemodelv3.h5', custom_objects={'L1Dist':L1Dist})
        # setup video capture 
        self.capture=cv2.VideoCapture(0)
        Clock.schedule_interval(self.update,1.0/33.0)

        return layout
    
    # run continuously to get webcam feed
    def update(self, *args):
        
        # read frame from opencv
        ret,frame=self.capture.read()
        frame=frame[120:120+250,200:200+250,:]

        # flip horizontal and convert image to texture
        buf= cv2.flip(frame,0).tostring()
        img_texture=Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        img_texture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
        self.webcam.texture=img_texture


    # load image from file and convert to 100 X 100
    def preprocess(self,path):
        # read the image from the path
        byte_img=tf.io.read_file(path)

        # load the image
        img= tf.io.decode_jpeg(byte_img)

        # preprocessing - resizing the image to 100X100
        img=tf.image.resize(img,(100,100))

        # scale image to be between 0 and 1
        img=img/255.0

        return img
    
    # Verification function
    def verify(self,*args):

        detection_threshold=0.9
        verification_threshold=0.7

        #capture input image
        SAVE_PATH=os.path.join('application_data','input_images','input_image.jpg')
        ret,frame=self.capture.read()
        frame=frame[120:120+250,200:200+250,:]
        cv2.imwrite(SAVE_PATH, frame)

        results=[]
        for image in os.listdir(os.path.join('application_data','verification_images')):
            input_img=self.preprocess(os.path.join('application_data','input_images','input_image.jpg'))
            validation_img= self.preprocess(os.path.join('application_data','verification_images',image))
            
            result=self.model.predict(list(np.expand_dims([input_img,validation_img],axis=1)))
            results.append(result)
            
        #detection_threshold:  Metric above which a prediction is considered positive (50%)
        detection=np.sum(np.array(results)>detection_threshold)
        
        # Verfification threshold: proportion of positive predictions/total positive samples
        verification=detection/len(os.listdir(os.path.join('application_data','verification_images')))
        verified=verification>verification_threshold
        
        # set verification text
        self.verification_label.text='Verified' if verified==True else 'Unverified'

        #log details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified

if __name__=='__main__':
    CamApp().run()