#NAME:BHANUSHRAY GUPTA
#DATA SCIENCE TASK 1
#BHARAT INTERN

import tkinter as tk
from tkinter import *
import tkinter.messagebox
from PIL import Image, ImageTk
import DATA_PROCESSING
import MODELS_MLDS
import pandas as pd

class Root(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("1000x600+0+0")
        self.title("TITANIC CLASSIFICATION SYSTEM")
        self.configure(background="#dbd8d7")
        self.frames = {}
        
        container = tk.Frame(self, background="#dbd8d7")
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        for F in (StartPage, Page1, Page2, Page3, Page4):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame(StartPage)
        
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, background="#dbd8d7")
        
        label_title = tk.Label(self, text="TITANIC CLASSIFICATION SYSTEM", font=("Elephant", 28), bg="#00ff00",borderwidth=2)
        label_title.pack(pady=50)

        data_analysis_label = tk.Label(self, text="ANALYZE DATA", font=("Arial Bold", 14), bg="#dbd8d7")
        data_analysis_label.place(x=600, y=120)

        buttonA = tk.Button(self, text="TRAIN DISTRIBUTION", command=lambda: MODELS_MLDS.trainclassDistr(MODELS_MLDS.train_df), borderwidth=2, relief="groove", bg="#e8e8e8")
        buttonA.place(x=600, y=170, width=200, height=30)

        buttonB = tk.Button(self, text="FARE SURVIVAL MEAN", command=lambda: MODELS_MLDS.trainMeanFareSurvival(MODELS_MLDS.train_df), borderwidth=2, relief="groove", bg="#e8e8e8")
        buttonB.place(x=600, y=220, width=200, height=30)

        buttonC = tk.Button(self, text="SURVIVAL CLASS", command=lambda: MODELS_MLDS.trainClassSurvival(MODELS_MLDS.train_df), borderwidth=2, relief="groove", bg="#e8e8e8")
        buttonC.place(x=600, y=270, width=200, height=30)

        data_analysis_label = tk.Label(self, text="DATA SCIENCE & MACHINE LEARNING MODELS", font=("Arial Bold", 14), bg="#dbd8d7")
        data_analysis_label.place(x=50, y=120)

        button1 = tk.Button(self, text="LOGISTIC REGRESSION", command=lambda: controller.show_frame(Page1), borderwidth=2, relief="groove", bg="#e8e8e8")
        button1.place(x=50, y=170, width=200, height=30)

        button2 = tk.Button(self, text="KNN (K-NEAREST NEIGHBOR) ALGORITHM", command=lambda: controller.show_frame(Page2), borderwidth=2, relief="groove", bg="#e8e8e8")
        button2.place(x=50, y=210, width=300, height=30)

        button3 = tk.Button(self, text="DECISION TREE", command=lambda: controller.show_frame(Page3), borderwidth=2, relief="groove", bg="#e8e8e8")
        button3.place(x=50, y=250, width=200, height=30)

        button4 = tk.Button(self, text="NAIVE BAYES ALGORITHM", command=lambda: controller.show_frame(Page4), borderwidth=2, relief="groove", bg="#e8e8e8")
        button4.place(x=50, y=290, width=200, height=30)
        
        label_made = tk.Label(self, text="MADE BY: BHANUSHRAY GUPTA", font=("Arial Bold", 25), bg="#dbd8d7")
        label_made.place(x=60,y=500)

class Page1(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        bg_image = Image.open('Images/bg2.gif')
        bg_image = bg_image.resize((1800, 780))
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(self, image=self.bg_photo)
        bg_label.place(x=0, y=0)

        lr_image = Image.open('Images/LogRegression.png')
        lr_image = lr_image.resize((700, 500))
        self.lr_photo = ImageTk.PhotoImage(lr_image)
        lr_label = tk.Label(self, image=self.lr_photo)
        lr_label.place(x=0, y=0)

        prediction_text = MODELS_MLDS.logRegression(MODELS_MLDS.train1, MODELS_MLDS.train2, MODELS_MLDS.test)
        prediction_text_area = tk.Text(self, font=("Arial", 12), height=20, width=50, wrap=tk.WORD)
        prediction_text_area.insert(tk.END, prediction_text)
        prediction_text_area.configure(state='disabled')

        prediction_scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=prediction_text_area.yview)
        prediction_text_area.configure(yscrollcommand=prediction_scrollbar.set)
        prediction_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        prediction_text_area.place(x=800, y=100)

        pred_label = tk.Label(self, text="PREDICTION", font=("Arial", 18, "bold"))
        pred_label.place(x=800, y=50)

        pred_hover_label = tk.Label(self, text="MODEL ACCURACY", font=("Arial", 12))
        pred_hover_label.place(x=70, y=820)
        
        graph_button = tk.Button(self, text="GRAPHICAL REPRESENTATION OF PREDICTION", font=("Arial", 14), command=lambda: MODELS_MLDS.groupPlot(MODELS_MLDS.predLog))
        graph_button.place(x=70, y=600)

        
        back_button = tk.Button(self, text="START PAGE", font=("Arial", 14), command=lambda: controller.show_frame(StartPage))
        back_button.place(x=70, y=700)


class Page2(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        load1 = Image.open('Images/bg2.gif')
        load1 = load1.resize((1800, 780))
        render1 = ImageTk.PhotoImage(load1)
        bg_label = tk.Label(self, image=render1)
        bg_label.image = render1
        bg_label.place(x=0, y=0)
        
        load2 = Image.open('Images/knn.png')
        load2 = load2.resize((700, 500))
        render2 = ImageTk.PhotoImage(load2)
        model_label = tk.Label(self, image=render2)
        model_label.image = render2
        model_label.place(x=70, y=60)

        pred_textarea = tk.Text(self, height=20, width=30, wrap=tk.WORD, font=("Arial", 12))
        pred_textarea.insert(tk.END, MODELS_MLDS.KNN(MODELS_MLDS.train1, MODELS_MLDS.train2, MODELS_MLDS.test))
        pred_textarea.config(state="disabled")
        pred_textarea_scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=pred_textarea.yview)
        pred_textarea.configure(yscrollcommand=pred_textarea_scrollbar.set)
        pred_textarea_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        pred_textarea.place(x=800, y=100)


        pred_label = tk.Label(self, text="PREDICTION", font=("Arial", 18, "bold"))
        pred_label.place(x=800, y=50)

        pred_hover_label = tk.Label(self, text="MODEL ACCURACY", font=("Arial", 12))
        pred_hover_label.place(x=70, y=820)

        back_button = tk.Button(self, text="START PAGE", font=("Arial", 14), command=lambda: controller.show_frame(StartPage))
        back_button.place(x=70, y=700)

        graph_button = tk.Button(self, text="GRAPHICAL REPRESENTATION OF PREDICTION", font=("Arial", 14), command=lambda: MODELS_MLDS.groupPlot(MODELS_MLDS.predK))
        graph_button.place(x=70, y=600)



class Page3(tk.Frame):
        def __init__(self, parent, controller):
                tk.Frame.__init__(self, parent)
                
                load1 = Image.open('Images/bg2.gif')
                load1 = load1.resize((1800, 780))
                render1 = ImageTk.PhotoImage(load1)
                img1 = tk.Label(self, image = render1)
                img1.image = render1
                img1.place(x = 0, y = 0)

                load2 = Image.open('Images/dt.png')
                load2 = load2.resize((700, 500))
                render2 = ImageTk.PhotoImage(load2)
                img2 = tk.Label(self, image = render2)
                img2.image = render2
                img2.place(x = 0, y = 0)

                textArea = tk.Text(self, height = 20, width = 15, wrap = tk.WORD)
                textArea.insert(tk.END, MODELS_MLDS.dTree(MODELS_MLDS.train1, MODELS_MLDS.train2, MODELS_MLDS.test))
                textArea.configure(font=("Arial",12))

                scroller = tk.Scrollbar(self, orient= tk.VERTICAL)
                scroller.config(command = textArea.yview)
                textArea.configure(yscrollcommand= scroller.set)
                scroller.pack(side = tk.RIGHT, fill = tk.Y)
                textArea.place(x = 800, y = 100)

                pred_label = tk.Label(self, text="PREDICTION", font=("Arial", 18, "bold"))
                pred_label.place(x=800, y=50)

                pred_hover_label = tk.Label(self, text="MODEL ACCURACY", font=("Arial", 12))
                pred_hover_label.place(x=70, y=820)

                back_button = tk.Button(self, text="START PAGE", font=("Arial", 14), command=lambda: controller.show_frame(StartPage))
                back_button.place(x=70, y=700)

                graph_button = tk.Button(self, text="GRAPHICAL REPRESENTATION OF PREDICTION", font=("Arial", 14), command=lambda: MODELS_MLDS.groupPlot(MODELS_MLDS.predTree))
                graph_button.place(x=70, y=600)


class Page4(tk.Frame):
        def __init__(self, parent, controller):
                tk.Frame.__init__(self, parent)

                load1 = Image.open('Images/bg2.gif')
                load1 = load1.resize((1800, 780))
                render1 = ImageTk.PhotoImage(load1)
                img1 = tk.Label(self, image = render1)
                img1.image = render1
                img1.place(x = 0, y = 0)

                load2 = Image.open('Images/NVB.png')
                load2 = load2.resize((700, 500))
                render2 = ImageTk.PhotoImage(load2)
                img2 = tk.Label(self, image = render2)
                img2.image = render2
                img2.place(x = 0, y = 0)

                textArea = tk.Text(self, height = 20, width = 15, wrap = tk.WORD)
                textArea.insert(tk.END, MODELS_MLDS.gNaiveBayes(MODELS_MLDS.train1, MODELS_MLDS.train2, MODELS_MLDS.test))
                textArea.configure(font=("Arial",12))

                scroller = tk.Scrollbar(self, orient= tk.VERTICAL)
                scroller.config(command = textArea.yview)
                textArea.configure(yscrollcommand= scroller.set)
                scroller.pack(side = tk.RIGHT, fill = tk.Y)
                textArea.place(x = 800, y = 100)

                pred_label = tk.Label(self, text="PREDICTION", font=("Arial", 18, "bold"))
                pred_label.place(x=800, y=50)

                pred_hover_label = tk.Label(self, text="MODEL ACCURACY", font=("Arial", 12))
                pred_hover_label.place(x=70, y=820)

                back_button = tk.Button(self, text="START PAGE", font=("Arial", 14), command=lambda: controller.show_frame(StartPage))
                back_button.place(x=70, y=700)

                graph_button = tk.Button(self, text="GRAPHICAL REPRESENTATION OF PREDICTION", font=("Arial", 14), command=lambda: MODELS_MLDS.groupPlot(MODELS_MLDS.predBayes))
                graph_button.place(x=70, y=600)


                
display = Root()
display.mainloop()
