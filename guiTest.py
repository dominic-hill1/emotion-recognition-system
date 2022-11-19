import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.title("Emotion recognition system")
        self.geometry("500x400")

        self.frames = {}

        for F in (Menu, LiveApplication, Analysis):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(Menu)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()


class Menu(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        header = tk.Label(self, text='Menu')
        header.config(font=("Courier", 40))
        header.pack(pady=10, padx=10)
        
        button = tk.Button(self, text="Visit Page 1",
                            command=lambda: controller.show_frame(LiveApplication))
        button.pack()

        button2 = tk.Button(self, text="Visit Page 2",
                            command=lambda: controller.show_frame(Analysis))
        button2.pack()

class LiveApplication(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        header = tk.Label(self, text='Live application')
        header.config(font=("Courier", 40))
        header.pack(pady=10, padx=10)

        
        button = tk.Button(self, text="Visit menu",
                            command=lambda: controller.show_frame(Menu))
        button.pack()

        button2 = tk.Button(self, text="Visit data analysis",
                            command=lambda: controller.show_frame(Analysis))
        button2.pack()

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


