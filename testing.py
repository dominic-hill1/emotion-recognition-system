import tkinter as tk
import cv2
from PIL import Image, ImageTk
 
cap = cv2.VideoCapture(0)
capWidth = cap.get(3)
capHeight = cap.get(4)

mainWindow = tk.Tk(screenName="Live application")
# mainWindow.resizable(width=False, height=False)
mainWindow.bind('<Escape>', lambda e: mainWindow.quit())
vidMain = tk.Label(mainWindow)
vidMain.pack(side='top', anchor='nw')

def show_frame():
 
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
 
    prevImg = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=prevImg)
    vidMain.imgtk = imgtk
    vidMain.configure(image=imgtk)
    vidMain.after(10, show_frame)
 
show_frame()
mainWindow.mainloop()


