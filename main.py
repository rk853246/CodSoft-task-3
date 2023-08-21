import tkinter as tk
import cv2

class FaceDetection:
    def __init__(self,root):
        self.root=root
        self.root.title("Face Detection and Recognition Application")
        self.root.geometry('450x500')
        self.button1=tk.Button(self.root,text="Face Detect",font=("Helvetica",15 ),command=lambda:self.FaceDetect())
        self.button1.place(x=167,y=170)
        self.button2=tk.Button(self.root,text="Face Recognize",font=("Helvetica",15 ))
        self.button2.place(x=150,y=220)


    def FaceDetect(self):
        
        self.Trained_Face_Data=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        self.Web_Camera=cv2.VideoCapture(0)

        while True:
            self.frame_read,self.frame=self.Web_Camera.read()
            self.grayscaleImg=cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
            self.face_coordinates = self.Trained_Face_Data.detectMultiScale(self.grayscaleImg) 
            for(x , y , w , h) in self.face_coordinates:
                cv2.rectangle(self.frame,(x , y) , (x+w , y+h) , (0, 256, 0) , 2)
            
            cv2.imshow('My Frame',self.frame)
            self.key=cv2.waitKey(1)

            if  self.key== 81 or self.key == 113:
                cv2.destroyWindow('My Frame')
                break
            
        self.Web_Camera.release()
    

    


if __name__=="__main__":
    root=tk.Tk()
    object=FaceDetection(root)
    root.mainloop()
