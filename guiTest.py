import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.title("Emotion recognition system")
        self.geometry("500x500")


        self.frames = {}

        for F in (Menu, LiveApplication, Analysis):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(Menu)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        if cont == LiveApplication:
            self.geometry("1000x600")
        else:
            self.geometry("400x200")

class Menu(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        header = tk.Label(self, text='Menu')
        header.config(font=("Courier", 40))
        header.pack(pady=10, padx=10)
        
        button = tk.Button(self, text="Visit Live app",
                            command=lambda: controller.show_frame(LiveApplication))
        button.pack()

        button2 = tk.Button(self, text="Visit Data analysis",
                            command=lambda: controller.show_frame(Analysis))
        button2.pack()

class LiveApplication(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        for i in range(1, 8):
            self.rowconfigure(i, weight=1)
        
        self.columnconfigure(1, weight=2)
        self.columnconfigure(2, weight=1)

        header = tk.Label(self, text='Live application')
        header.config(font=("Courier", 40))
        header.grid(row=0, sticky='ew')

        subheader = tk.Label(self, text='Emotions:')
        subheader.config(font=("Courier", 25))
        subheader.grid(row=0, column=2, sticky='ew')

        for rowNum, emotion in enumerate(emotions):
            emotion = ttk.Progressbar(
                self,
                orient='horizontal',
                mode='determinate',
                length=280,
                value=40,
            )

            emotion.grid(column=2, row=rowNum+1, padx=30, pady=20)

        vidMain = tk.Label(self)
        vidMain.grid(row=1, sticky='nw', rowspan=7)

        button = tk.Button(self, text="Visit menu",
                            command=lambda: controller.show_frame(Menu))
        button.grid(row=8, sticky='sw')




        # model = self.load_model("20220821-12521661086371-full-image-set-mobilenetv2-Adam.h5")

        # # Load cascade classifier into memory
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        cap = cv2.VideoCapture(0)
        self.delay_counter = 1
        self.grey_face = False
    

    

        def display_frame():

 
            _, frame = cap.read() # take frame from webcam
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # create image to show to user
            greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to greyscale

            # # Pass into classifier
            # faces = face_cascade.detectMultiScale(
            #     greyscale,
            #     scaleFactor=1.2,
            #     minNeighbors=5,
            #     minSize=(50,50)
            # )
        
            # for (x, y, w, h) in faces: # Iterate over faces detected 
            #     # self.overlay = frame.copy()
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2) # draw rectangle around face
            #     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # create new image to show to user

            #     self.grey_face = greyscale[y:y+h, x:x+w] # crop face
            
                
            # if self.delay_counter % 60 == 0 and hasattr(self.grey_face, 'shape'):
            #     try:
            #         self.grey_face = self.process_image(self.grey_face)
            #         predictions = model.predict(np.array([self.grey_face]))
            #         print(emotions[np.argmax(predictions)])
            #     except:
            #         pass

            # self.delay_counter += 1
    
            prevImg = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=prevImg)
            vidMain.imgtk = imgtk
            vidMain.configure(image=imgtk)
            vidMain.after(10, display_frame)

            
 
        display_frame()


    def load_model(self, model_path):
        """
        Loads a saved model from a specified path.
        """
        print(f"Loading saved model from: {model_path}")
        model = tf.keras.models.load_model(model_path,
                                            custom_objects={"KerasLayer": hub.KerasLayer})
        return model

    def process_image(self, greyscale_image):
        """
        Formats an image, ready to be passed into neural network
        """
        # Converts 1 greyscale channel into 3 identical RGB channels
        greyscale_image = cv2.cvtColor(greyscale_image,cv2.COLOR_GRAY2RGB)
        # Changes values from 0-255 to 0-1
        normalized_image = tf.image.convert_image_dtype(greyscale_image, tf.float32)
        # Changes size of image to 244x244 pixels
        formatted_image = tf.image.resize(normalized_image, size=[224, 224])

        return formatted_image

    


class Analysis(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        header = tk.Label(self, text='Data Analysis')
        header.config(font=("Courier", 40))
        header.pack(pady=10, padx=10)

        
        button = tk.Button(self, text="Visit menu",
                            command=lambda: controller.show_frame(Menu))
        button.pack()

        button2 = tk.Button(self, text="Visit live application",
                            command=lambda: controller.show_frame(LiveApplication))
        button2.pack()

if __name__ == "__main__":
    app = App()
    app.mainloop()


