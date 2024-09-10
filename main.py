# Import libraries
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
    """
    App class to control whole of GUI and its functionality.
    Subclasses:
        - Menu
        - LiveApplication
        - Analysis
    """
    def __init__(self):
        super().__init__()

        # Create a container to hold all widgets
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)

        # Configure geometry manager for container
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.title("Emotion recognition system")
        self.geometry("500x500")


        self.frames = {}
        # Create a container for each section of program.
        for F in (Menu, LiveApplication, Analysis):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(Menu)

    def show_frame(self, cont):
        # Raise frame to top of stack so it is displayed to user
        frame = self.frames[cont]
        frame.tkraise()

        # Change the size of the screen depending on program
        if cont == LiveApplication:
            self.geometry("1000x600")
        elif cont == Analysis:
            # self.geometry("1920x1080")
            self.geometry("1000x600")
        else:
            self.geometry("400x200")

class Menu(tk.Frame):
    """
    A class to control the menu section of the program
    The program opens with the menu and it welcomes the user
    The menu provides links to the live application and data analysis
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # Create a header for text
        header = tk.Label(self, text='Menu')
        header.config(font=("Courier", 40))
        header.pack(pady=10, padx=10)
        
        # Create links to other parts of the program
        button = tk.Button(self, text="Visit Live app",
                            command=lambda: controller.show_frame(LiveApplication))
        button.pack()

        button2 = tk.Button(self, text="Visit Data analysis",
                            command=lambda: controller.show_frame(Analysis))
        button2.pack()

class LiveApplication(tk.Frame):
    """
    A class to control the live application section of the program
    This class manages:
        - Webcam input
        - Emotion prediction
        - Visual output to user
        - Storage of emotions
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # List of all possible emotions
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # Configure rows and columns for geometry manager
        for i in range(1, 16):
            self.rowconfigure(i, weight=1)
        
        self.columnconfigure(1, weight=2)
        self.columnconfigure(2, weight=1)
       
        # Create heading and subheading for GUI
        header = tk.Label(self, text='Emotion Recognition')
        header.config(font=("Courier", 30))
        header.grid(row=0, sticky='ew')

        subheader = tk.Label(self, text='Emotions:')
        # subheader.config(font=("Courier", 25))
        subheader.config(font=("Courier", 20))
        subheader.grid(row=0, column=2, sticky='ew')

        # Create progress bars to represent amounts of emotions to user

        angerLabel = tk.Label(self, text='Anger') # Instantiate a label for emotion
        angerLabel.grid(column=2, row=1, padx=30, pady=5, sticky='nsew') # Put label in correct position

        self.angerBar = ttk.Progressbar( # Instantiate a progress bar for emotion
            self,
            orient='horizontal',
            mode='determinate',
            length=280,
            value=0,
        )
        self.angerBar.grid(column=2, row=2, padx=30, pady=5, sticky="s") # put progress bar in correct position

        # Repeat for each emotion

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

        # Create output for video
        vidMain = tk.Label(self)
        vidMain.grid(row=1, sticky='nw', rowspan=15)

        # Create button to return to menu
        button = tk.Button(self, text="Visit menu",
                            command=lambda: controller.show_frame(Menu))
        button.grid(row=16, sticky='sw')



        # Load saved model
        model = self.load_model("20220821-12521661086371-full-image-set-mobilenetv2-Adam.h5")

        # Load cascade classifier into memory
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Begin video capture
        cap = cv2.VideoCapture(0)
        self.delay_counter = 1 # Use delay counter to decrease number of predictions
        self.grey_face = False

        def display_frame():
            _, frame = cap.read() # Take frame from webcam
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # Create image to show to user
            greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to greyscale

            # Pass into classifier
            faces = face_cascade.detectMultiScale(
                greyscale,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(50,50)
            )
        
            for (x, y, w, h) in faces: # Iterate over faces detected 
                # self.overlay = frame.copy()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2) # Draw rectangle around face
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # Create new image to show to user

                self.grey_face = greyscale[y:y+h, x:x+w] # Crop face
            
                
            if self.delay_counter % 60 == 0 and hasattr(self.grey_face, 'shape'):
                try:
                    self.grey_face = self.process_image(self.grey_face) # Format image
                    predictions = model.predict(np.array([self.grey_face])) # Make prediction on image
                    # Update prediction progress bars to user
                    self.angerBar['value'] = predictions[0][0] * 100
                    self.disgustBar['value'] = predictions[0][1] * 100
                    self.fearBar['value'] = predictions[0][2] * 100
                    self.happyBar['value'] = predictions[0][3] * 100
                    self.neutralBar['value'] = predictions[0][4] * 100
                    self.sadBar['value'] = predictions[0][5] * 100
                    self.surpriseBar['value'] = predictions[0][6] * 100
                    # Append maxmimum recorded value to records.csv file
                    record = pd.DataFrame([[emotions[np.argmax(predictions)], datetime.datetime.now()]],
                                            columns=["emotion", "datetime"])
                    record.to_csv('records.csv', mode='a', index=False, header=False)
                except:
                    pass

            self.delay_counter += 1

            # Show video feed to user in GUI
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

        # Configure rows and columns for geometry manager
        for i in range(1, 16):
            self.rowconfigure(i, weight=1)
        
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        # Create header for section
        header = tk.Label(self, text='Data Analysis')
        header.config(font=("Courier", 40))
        header.grid(column=1, row=1, pady=10, padx=10, columnspan=2)

        # Initialise graphs
        self.figure, (self.ax0, self.ax1) = plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(5,4),
                            sharex=False,
                            sharey=True)
        # Set labels
        self.ax0.set_xlabel('datetime')
        self.ax0.set_title('Emotion over time')

        self.ax1.set_xlabel('datetime')
  
        # Add padding to bottom of graph
        plt.gcf().subplots_adjust(bottom=0.15)

        # Show graph as a widget on the GUI
        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graphWidget = self.graph.get_tk_widget()
        self.graphWidget.grid(row=2, column=1, padx=0, pady=40, rowspan=15)

        # Create button to go to menu
        menuButton = tk.Button(self, text="Visit menu",
                            command=lambda: controller.show_frame(Menu))
        menuButton.grid(column=2, row=2, sticky='n')

        # Create button to go to live application
        liveButton = tk.Button(self, text="Visit live application",
                            command=lambda: controller.show_frame(LiveApplication))
        liveButton.grid(column=2, row=3, sticky='n')

        # Create lists for items on drop down menus
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
        
        # Create subtitle and drop down menu for first emotion
        firstHeader = tk.Label(self, text="Pick your first emotion:")
        self.firstEmotion = tk.StringVar()
        self.firstEmotion.set("Happy")
        dropdown1 = tk.OptionMenu(self, self.firstEmotion, *emotionOptions)

        # Create subtitle and drop down menu for second emotion
        secondHeader = tk.Label(self, text="Pick your second emotion:")
        self.secondEmotion = tk.StringVar()
        self.secondEmotion.set("Sad")
        dropdown2 = tk.OptionMenu(self, self.secondEmotion, *emotionOptions)
        
        # Create subtitle and drop down menu for time format
        timeHeader = tk.Label(self, text="Pick a time format")
        self.selectedTime = tk.StringVar()
        self.selectedTime.set("Day/Month/Year")
        dropdownTime = tk.OptionMenu(self, self.selectedTime, *timeOptions)
        
        # Create a button which will call the 'updateGraphs' function
        updateButton = tk.Button(self, text="Update graphs", command=self.updateGraphs)
        
        # Arrange all elements on grid
        firstHeader.grid(column=2, row=4)
        dropdown1.grid(column=2, row=5)
        secondHeader.grid(column=2, row=6)
        dropdown2.grid(column=2, row=7)
        timeHeader.grid(column=2, row=8)
        dropdownTime.grid(column=2, row=9)
        updateButton.grid(column=2, row=10)


    def updateGraphs(self):
        """
        Plots and formats a graph to display to the user
        """
        # Import csv file into pandas DataFrame
        self.recordsDF = pd.read_csv("records.csv")
        self.recordsDF["datetime"] = pd.to_datetime(self.recordsDF["datetime"]) # Change data type
        # Format the DataFrame to select the correct time format selected by the user
        if self.selectedTime.get() == "Day/Month/Year":
            self.recordsDF['datetime'] = self.recordsDF['datetime'].dt.strftime("%d/%m/%y")
        elif self.selectedTime.get() == "Hour:Minute":
            self.recordsDF['datetime'] = self.recordsDF['datetime'].dt.strftime("%H:%M")
        elif self.selectedTime.get() == "Day":
            self.recordsDF['datetime'] = self.recordsDF['datetime'].dt.strftime("%d")
        elif self.selectedTime.get() == "Month":
            self.recordsDF['datetime'] = self.recordsDF['datetime'].dt.strftime("%m")
        elif self.selectedTime.get() == "Year":
            self.recordsDF['datetime'] = self.recordsDF['datetime'].dt.strftime("%y")
        
        # Create a first dataframe which creates a copy of recordsDF but with only one emotion
        if self.firstEmotion.get() == "Happy":
            firstDF = self.recordsDF[self.recordsDF["emotion"]=="happy"]
        elif self.firstEmotion.get() == "Sad":
            firstDF = self.recordsDF[self.recordsDF["emotion"]=="sad"]
        elif self.firstEmotion.get() == "Angry":
            firstDF = self.recordsDF[self.recordsDF["emotion"]=="angry"]
        elif self.firstEmotion.get() == "Neutral":
            firstDF = self.recordsDF[self.recordsDF["emotion"]=="neutral"]
        elif self.firstEmotion.get() == "Disgust":
            firstDF = self.recordsDF[self.recordsDF["emotion"]=="disgust"]
        elif self.firstEmotion.get() == "Fear":
            firstDF = self.recordsDF[self.recordsDF["emotion"]=="fear"]
        elif self.firstEmotion.get() == "Surprise":
            firstDF = self.recordsDF[self.recordsDF["emotion"]=="surprise"]

        # Group the data by the datetime
        groupedTime = firstDF.groupby('datetime')
        groupedTime.head()
        # Count the number of emotion detections at that time. 
        firstCounts = groupedTime["emotion"].count()

        # Create a second dataframe which creates a copy of recordsDF but with only one emotion
        if self.secondEmotion.get() == "Happy":
            secondDF = self.recordsDF[self.recordsDF["emotion"]=="happy"]
        elif self.secondEmotion.get() == "Sad":
            secondDF = self.recordsDF[self.recordsDF["emotion"]=="sad"]
        elif self.secondEmotion.get() == "Angry":
            secondDF = self.recordsDF[self.recordsDF["emotion"]=="angry"]
        elif self.secondEmotion.get() == "Neutral":
            secondDF = self.recordsDF[self.recordsDF["emotion"]=="neutral"]
        elif self.secondEmotion.get() == "Disgust":
            secondDF = self.recordsDF[self.recordsDF["emotion"]=="disgust"]
        elif self.secondEmotion.get() == "Fear":
            secondDF = self.recordsDF[self.recordsDF["emotion"]=="fear"]
        elif self.secondEmotion.get() == "Surprise":
            secondDF = self.recordsDF[self.recordsDF["emotion"]=="surprise"] 

        # Group the data by the datetime
        groupedTime = secondDF.groupby('datetime')
        groupedTime.head()
        # Count the number of emotion detections at that time. 
        secondCounts = groupedTime["emotion"].count()

        # Destroy existing graph widget to make space for new one
        self.graphWidget.destroy()
        # Intialise new graph
        self.figure, (self.ax0, self.ax1) = plt.subplots(nrows=2,
                                  ncols=1,
                                  figsize=(5,4),
                                  sharex=False,
                                  sharey=True)

        # Set title for graphs
        self.ax0.set_title('Emotion over time')

        # Change X-axes for graphs depending on time format chosen
        if self.selectedTime.get() == "Day/Month/Year":
            self.ax0.set_xlabel('Date (Day/Month/Year)')
            self.ax1.set_xlabel('Date (Day/Month/Year)')
        elif self.selectedTime.get() == "Hour:Minute":
            self.ax0.set_xlabel('Time (Hour:Min)')
            self.ax1.set_xlabel('Time (Hour:Min)')
        elif self.selectedTime.get() == "Day":
            self.ax0.set_xlabel('Day')
            self.ax1.set_xlabel('Day')
        elif self.selectedTime.get() == "Month":
            self.ax0.set_xlabel('Month')
            self.ax1.set_xlabel('Month')
        elif self.selectedTime.get() == "Year":
            self.ax0.set_xlabel('Year')
            self.ax1.set_xlabel('Year')
        
        # Set Y-axis for first graph depending on emotion being plotted
        if self.firstEmotion.get() == "Happy":
            self.ax0.set_ylabel("Happiness")
        elif self.firstEmotion.get() == "Sad":
            self.ax0.set_ylabel("Sadness")
        elif self.firstEmotion.get() == "Angry":
            self.ax0.set_ylabel("Anger")
        elif self.firstEmotion.get() == "Neutral":
            self.ax0.set_ylabel("Neutrality")
        elif self.firstEmotion.get() == "Disgust":
            self.ax0.set_ylabel("Disgust")
        elif self.firstEmotion.get() == "Fear":
            self.ax0.set_ylabel("Fear")
        elif self.firstEmotion.get() == "Surprise":
            self.ax0.set_ylabel("Surprise")
        
        # Set Y-axis for second graph depending on emotion being plotted
        if self.secondEmotion.get() == "Happy":
            self.ax1.set_ylabel("Happiness")
        elif self.secondEmotion.get() == "Sad":
            self.ax1.set_ylabel("Sadness")
        elif self.secondEmotion.get() == "Angry":
            self.ax1.set_ylabel("Anger")
        elif self.secondEmotion.get() == "Neutral":
            self.ax1.set_ylabel("Neutrality")
        elif self.secondEmotion.get() == "Disgust":
            self.ax1.set_ylabel("Disgust")
        elif self.secondEmotion.get() == "Fear":
            self.ax1.set_ylabel("Fear")
        elif self.secondEmotion.get() == "Surprise":
            self.ax1.set_ylabel("Surprise") 

        # Plot the data from above
        self.ax0.plot(firstCounts, color="g")
        self.ax1.plot(secondCounts, color="r")

        # Give the the graphs extra padding on the GUI
        plt.gcf().subplots_adjust(bottom=0.15)

        # Create GUI widget for the graph and put this graph on the grid
        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graphWidget = self.graph.get_tk_widget()
        self.graphWidget.grid(row=2, column=1, padx=0, pady=40, rowspan=15)


# Run program
if __name__ == "__main__":
    app = App() # Initalise app object
    app.mainloop() # Initalise GUI



