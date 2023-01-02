import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        elif cont == Analysis:
            self.geometry("1920x1080")
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

        for i in range(1, 16):
            self.rowconfigure(i, weight=1)
        
        self.columnconfigure(1, weight=2)
        self.columnconfigure(2, weight=1)

        # header = tk.Label(self, text='Live application')
        # header.config(font=("Courier", 40))        
        header = tk.Label(self, text='Emotion Recognition')
        header.config(font=("Courier", 30))
        header.grid(row=0, sticky='ew')

        subheader = tk.Label(self, text='Emotions:')
        # subheader.config(font=("Courier", 25))
        subheader.config(font=("Courier", 20))
        subheader.grid(row=0, column=2, sticky='ew')

        angerLabel = tk.Label(self, text='Anger') # instantiate a label for emotion
        angerLabel.grid(column=2, row=1, padx=30, pady=5, sticky='nsew') # put label in correct position

        self.angerBar = ttk.Progressbar( # instantiate a progress bar for emotion
            self,
            orient='horizontal',
            mode='determinate',
            length=280,
            value=0,
        )
        self.angerBar.grid(column=2, row=2, padx=30, pady=5, sticky="s") # put progress bar in correct position

        disgustLabel = tk.Label(self, text='Disgust')
        disgustLabel.grid(column=2, row=3, padx=30, pady=5, sticky='nsew')

        self.disgustBar = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='determinate',
            length=280,
            value=0,
        )
        self.disgustBar.grid(column=2, row=4, padx=30, pady=5, sticky="s")

        fearLabel = tk.Label(self, text='Fear')
        fearLabel.grid(column=2, row=5, padx=30, pady=5, sticky='nsew')
        self.fearBar = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='determinate',
            length=280,
            value=0,
        )
        self.fearBar.grid(column=2, row=6, padx=30, pady=5, sticky="s")

        happyLabel = tk.Label(self, text='Happy')
        happyLabel.grid(column=2, row=7, padx=30, pady=5, sticky='nsew')
        self.happyBar = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='determinate',
            length=280,
            value=0,
        )
        self.happyBar.grid(column=2, row=8, padx=30, pady=5, sticky='s')

        neutralLabel = tk.Label(self, text='Neutral')
        neutralLabel.grid(column=2, row=9, padx=30, pady=5, sticky='nsew')
        self.neutralBar = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='determinate',
            length=280,
            value=0,
        )
        self.neutralBar.grid(column=2, row=10, padx=30, pady=5, sticky="s")

        sadLabel = tk.Label(self, text='Sad')
        sadLabel.grid(column=2, row=11, padx=30, pady=5, sticky='nsew')
        self.sadBar = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='determinate',
            length=280,
            value=0,
        )
        self.sadBar.grid(column=2, row=12, padx=30, pady=5, sticky="s")

        surpriseLabel = tk.Label(self, text='Surprise')
        surpriseLabel.grid(column=2, row=13, padx=30, pady=5, sticky='nsew')
        self.surpriseBar = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='determinate',
            length=280,
            value=0,
        )
        self.surpriseBar.grid(column=2, row=14, padx=30, pady=5, sticky="s")

        # TODO: change color of progress bars


        vidMain = tk.Label(self)
        vidMain.grid(row=1, sticky='nw', rowspan=15)

        button = tk.Button(self, text="Visit menu",
                            command=lambda: controller.show_frame(Menu))
        button.grid(row=16, sticky='sw')




        model = self.load_model("20220821-12521661086371-full-image-set-mobilenetv2-Adam.h5")

        # Load cascade classifier into memory
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        cap = cv2.VideoCapture(0)
        self.delay_counter = 1
        self.grey_face = False
    

    

        def display_frame():
            _, frame = cap.read() # take frame from webcam
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # create image to show to user
            greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to greyscale

            # Pass into classifier
            faces = face_cascade.detectMultiScale(
                greyscale,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(50,50)
            )
        
            for (x, y, w, h) in faces: # Iterate over faces detected 
                # self.overlay = frame.copy()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2) # draw rectangle around face
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # create new image to show to user

                self.grey_face = greyscale[y:y+h, x:x+w] # crop face
            
                
            if self.delay_counter % 60 == 0 and hasattr(self.grey_face, 'shape'):
                try:
                    self.grey_face = self.process_image(self.grey_face)
                    predictions = model.predict(np.array([self.grey_face]))
                    # print(predictions)
                    self.angerBar['value'] = predictions[0][0] * 100
                    self.disgustBar['value'] = predictions[0][1] * 100
                    self.fearBar['value'] = predictions[0][2] * 100
                    self.happyBar['value'] = predictions[0][3] * 100
                    self.neutralBar['value'] = predictions[0][4] * 100
                    self.sadBar['value'] = predictions[0][5] * 100
                    self.surpriseBar['value'] = predictions[0][6] * 100
                    # print(emotions[np.argmax(predictions)])
                    record = pd.DataFrame([[emotions[np.argmax(predictions)], datetime.datetime.now()]],
                                            columns=["emotion", "datetime"])
                    record.to_csv('records.csv', mode='a', index=False, header=False)
                except:
                    pass

            self.delay_counter += 1
    
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

        for i in range(1, 16):
            self.rowconfigure(i, weight=1)
        
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        header = tk.Label(self, text='Data Analysis')
        header.config(font=("Courier", 40))
        # header.pack(pady=10, padx=10)
        header.grid(column=1, row=1, pady=10, padx=10, columnspan=2)


        # self.updateGraphs()

        self.figure, (self.ax0, self.ax1) = plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(5,4),
                            sharex=False,
                            sharey=True)
        self.ax0.set_xlabel('datetime')
        self.ax0.set_ylabel('Happiness')
        self.ax0.set_title('Emotion over time')

        self.ax1.set_xlabel('datetime')
        self.ax1.set_ylabel('Sadness')
        # ax1.set_title('Sadness over time')

        # happyCounts.plot(label="happy", color='g')
        # self.ax0.plot(happyCounts, color="g")
        # self.ax1.plot(sadCounts, color="r")

        plt.gcf().subplots_adjust(bottom=0.15)

        # plt.xticks(rotation='vertical')
        # plt.xticks(rotation=45)


        # plt.legend()
        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graphWidget = self.graph.get_tk_widget()
        self.graphWidget.grid(row=2, column=1, padx=0, pady=40, rowspan=15)

        # self.recordsDF = pd.read_csv("records.csv")
        # self.recordsDF["datetime"] = pd.to_datetime(self.recordsDF["datetime"])

        menuButton = tk.Button(self, text="Visit menu",
                            command=lambda: controller.show_frame(Menu))
        menuButton.grid(column=2, row=2, sticky='n')

        liveButton = tk.Button(self, text="Visit live application",
                            command=lambda: controller.show_frame(LiveApplication))
        liveButton.grid(column=2, row=3, sticky='n')

        emotionOptions = ["Happy",
                          "Sad",
                          "Angry",
                          "Neutral",
                          "Disgust",
                          "Fear",
                          "Surprise"]

        timeOptions = ["Hour:Minute",
                       "Day/Month/Year",
                       "Day",
                       "Month",
                       "Year"]
        
        firstHeader = tk.Label(self, text="Pick your first emotion:")
        self.firstEmotion = tk.StringVar()
        self.firstEmotion.set("Happy")
        dropdown1 = tk.OptionMenu(self, self.firstEmotion, *emotionOptions)

        secondHeader = tk.Label(self, text="Pick your second emotion:")
        self.secondEmotion = tk.StringVar()
        self.secondEmotion.set("Sad")
        dropdown2 = tk.OptionMenu(self, self.secondEmotion, *emotionOptions)
        

        timeHeader = tk.Label(self, text="Pick a time format")
        self.selectedTime = tk.StringVar()
        self.selectedTime.set("Day/Month/Year")
        dropdownTime = tk.OptionMenu(self, self.selectedTime, *timeOptions)
        

        updateButton = tk.Button(self, text="Update graphs", command=self.updateGraphs)
        

        # self.updateGraphs()
        
        firstHeader.grid(column=2, row=4)
        dropdown1.grid(column=2, row=5)
        secondHeader.grid(column=2, row=6)
        dropdown2.grid(column=2, row=7)
        timeHeader.grid(column=2, row=8)
        dropdownTime.grid(column=2, row=9)
        updateButton.grid(column=2, row=10)


    def updateGraphs(self):
        self.recordsDF = pd.read_csv("records.csv")
        self.recordsDF["datetime"] = pd.to_datetime(self.recordsDF["datetime"])
        if self.selectedTime.get() == "Day/Month/Year":
            self.recordsDF['datetime'] = self.recordsDF['datetime'].dt.strftime("%d/%m/%y")
        elif self.selectedTime.get() == "Hour:Minute":
            self.recordsDF['datetime'] = self.recordsDF['datetime'].dt.strftime("%H:%M")
            print("1")
        elif self.selectedTime.get() == "Day":
            self.recordsDF['datetime'] = self.recordsDF['datetime'].dt.strftime("%d")
        elif self.selectedTime.get() == "Month":
            self.recordsDF['datetime'] = self.recordsDF['datetime'].dt.strftime("%m")
        elif self.selectedTime.get() == "Year":
            self.recordsDF['datetime'] = self.recordsDF['datetime'].dt.strftime("%y")
        
            


        sadDF = self.recordsDF[self.recordsDF["emotion"]=="sad"]
        groupedTime = sadDF.groupby('datetime')
        groupedTime.head()
        sadCounts = groupedTime["emotion"].count()

        happyDF = self.recordsDF[self.recordsDF["emotion"]=="happy"]
        groupedTime = happyDF.groupby('datetime')
        groupedTime.head()
        happyCounts = groupedTime["emotion"].count()

        self.graphWidget.destroy()

        self.figure, (self.ax0, self.ax1) = plt.subplots(nrows=2,
                                  ncols=1,
                                  figsize=(5,4),
                                  sharex=False,
                                  sharey=True)
        self.ax0.set_xlabel('datetime')
        self.ax0.set_ylabel('Happiness')
        self.ax0.set_title('Emotion over time')

        self.ax1.set_xlabel('datetime')
        self.ax1.set_ylabel('Sadness')
        # ax1.set_title('Sadness over time')

        # happyCounts.plot(label="happy", color='g')
        self.ax0.plot(happyCounts, color="g")
        self.ax1.plot(sadCounts, color="r")

        plt.gcf().subplots_adjust(bottom=0.15)

        # plt.xticks(rotation='vertical')
        # plt.xticks(rotation=45)


        # plt.legend()
        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graphWidget = self.graph.get_tk_widget()
        self.graphWidget.grid(row=2, column=1, padx=0, pady=40, rowspan=15)



if __name__ == "__main__":
    app = App()
    app.mainloop()

    # print(app.recordsDF)

    # app.recordsDF.to_csv("records.csv")


