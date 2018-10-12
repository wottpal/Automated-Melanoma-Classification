# -*- coding: utf-8 -*-
"""
GUI for prediciting single images.
"""

import cam, predict
from helpers import io, colors
from preprocessors import resize#, hair_removal, lesion_segmentation
from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import os
import pickle
from subprocess import call



class PredictionGUI:

    def __init__(self, top):        
        self.top = top
        top.geometry("680x600+309+23")
        top.title("Automated Melanoma Classification")
        top.configure(background="#d9d9d9")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")


        # Create Variables
        self.modelPath = StringVar()
        self.modelPathShown = StringVar()
        self.imagePath = StringVar()
        self.imagePathShown = StringVar()
        self.hairRemovalChecked = IntVar() 
        self.lesionSegmentChecked = IntVar()
        self.buildHeatmapsChecked = IntVar()

        self.predictButtonLabel = StringVar()
        self.predictButtonLabel.set("Preprocess & Predict")

        self.isLoading = False


        # Load Persisted Variable Content
        self.loadData()


        # Options Frame
        self.Frame1 = Frame(top)
        self.Frame1.configure(relief=SUNKEN)
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief=SUNKEN)
        self.Frame1.configure(background="#d9d9d9")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")
        self.Frame1.configure(width=640)
        self.Frame1.place(relx=0.03, rely=0.03, relheight=0.24, relwidth=0.94)

        # Model/Image Buttons & Labels
        self.ModelButton = Button(self.Frame1)
        self.ModelButton.configure(command=self.selectModelPath)
        self.ModelButton.configure(activebackground="#d9d9d9")
        self.ModelButton.configure(activeforeground="#000000")
        self.ModelButton.configure(background="#d9d9d9")
        self.ModelButton.configure(foreground="#000000")
        self.ModelButton.configure(highlightbackground="#d9d9d9")
        self.ModelButton.configure(highlightcolor="black")
        self.ModelButton.configure(text='''Select Model..''')
        self.ModelButton.place(relx=0.03, rely=0.14, height=22, width=137)

        self.ImageButton = Button(self.Frame1)
        self.ImageButton.configure(command=self.selectImagePath)
        self.ImageButton.configure(activebackground="#d9d9d9")
        self.ImageButton.configure(activeforeground="#000000")
        self.ImageButton.configure(background="#d9d9d9")
        self.ImageButton.configure(foreground="#000000")
        self.ImageButton.configure(highlightbackground="#d9d9d9")
        self.ImageButton.configure(highlightcolor="black")
        self.ImageButton.configure(text='''Select Image..''')
        self.ImageButton.place(relx=0.03, rely=0.35, height=22, width=137)

        self.ModelPathLabel = Label(self.Frame1)
        self.ModelPathLabel.configure(textvariable = self.modelPathShown)
        self.ModelPathLabel.configure(activebackground="#f9f9f9")
        self.ModelPathLabel.configure(activeforeground="black")
        self.ModelPathLabel.configure(anchor=W)
        self.ModelPathLabel.configure(background="#d9d9d9")
        self.ModelPathLabel.configure(foreground="#000000")
        self.ModelPathLabel.configure(highlightbackground="#d9d9d9")
        self.ModelPathLabel.configure(highlightcolor="black")
        self.ModelPathLabel.place(relx=0.26, rely=0.14, height=24, width=441)

        self.ImagePathLabel = Label(self.Frame1)
        self.ImagePathLabel.configure(textvariable = self.imagePathShown)
        self.ImagePathLabel.configure(activebackground="#f9f9f9")
        self.ImagePathLabel.configure(activeforeground="black")
        self.ImagePathLabel.configure(anchor=W)
        self.ImagePathLabel.configure(background="#d9d9d9")
        self.ImagePathLabel.configure(foreground="#000000")
        self.ImagePathLabel.configure(highlightbackground="#d9d9d9")
        self.ImagePathLabel.configure(highlightcolor="black")
        self.ImagePathLabel.configure(text='')
        self.ImagePathLabel.place(relx=0.26, rely=0.35, height=24, width=441)


        # Preprocess Options
        # self.HairRemovalCheck = Checkbutton(self.Frame1)
        # self.HairRemovalCheck.configure(variable=self.hairRemovalChecked)
        # self.HairRemovalCheck.configure(command=self.persistData)
        # self.HairRemovalCheck.configure(activebackground="#d9d9d9")
        # self.HairRemovalCheck.configure(activeforeground="#000000")
        # self.HairRemovalCheck.configure(anchor=W)
        # self.HairRemovalCheck.configure(background="#d9d9d9")
        # self.HairRemovalCheck.configure(foreground="#000000")
        # self.HairRemovalCheck.configure(highlightbackground="#d9d9d9")
        # self.HairRemovalCheck.configure(highlightcolor="black")
        # self.HairRemovalCheck.configure(justify=LEFT)
        # self.HairRemovalCheck.configure(text='''Remove Hair''')
        # self.HairRemovalCheck.place(relx=0.03, rely=0.7, relheight=0.15, relwidth=0.18)

        # self.LesionSegmentCheck = Checkbutton(self.Frame1)
        # self.LesionSegmentCheck.configure(variable=self.lesionSegmentChecked)
        # self.LesionSegmentCheck.configure(command=self.persistData)
        # self.LesionSegmentCheck.configure(activebackground="#d9d9d9")
        # self.LesionSegmentCheck.configure(activeforeground="#000000")
        # self.LesionSegmentCheck.configure(anchor=W)
        # self.LesionSegmentCheck.configure(background="#d9d9d9")
        # self.LesionSegmentCheck.configure(foreground="#000000")
        # self.LesionSegmentCheck.configure(highlightbackground="#d9d9d9")
        # self.LesionSegmentCheck.configure(highlightcolor="black")
        # self.LesionSegmentCheck.configure(justify=LEFT)
        # self.LesionSegmentCheck.configure(text='''Segment Lesion''')
        # self.LesionSegmentCheck.place(relx=0.218, rely=0.7, relheight=0.15, relwidth=0.21)

        self.BuildHeatmapsCheck = Checkbutton(self.Frame1)
        self.BuildHeatmapsCheck.configure(variable=self.buildHeatmapsChecked)
        self.BuildHeatmapsCheck.configure(command=self.persistData)
        self.BuildHeatmapsCheck.configure(activebackground="#d9d9d9")
        self.BuildHeatmapsCheck.configure(activeforeground="#000000")
        self.BuildHeatmapsCheck.configure(anchor=W)
        self.BuildHeatmapsCheck.configure(background="#d9d9d9")
        self.BuildHeatmapsCheck.configure(foreground="#000000")
        self.BuildHeatmapsCheck.configure(highlightbackground="#d9d9d9")
        self.BuildHeatmapsCheck.configure(highlightcolor="black")
        self.BuildHeatmapsCheck.configure(justify=LEFT)
        self.BuildHeatmapsCheck.configure(text='''Heatmaps''')
        # self.BuildHeatmapsCheck.place(relx=0.435, rely=0.7, relheight=0.15, relwidth=0.23)
        self.BuildHeatmapsCheck.place(relx=0.03, rely=0.7, relheight=0.15, relwidth=0.23)


        # Predict Button
        self.PredictButton = Button(self.Frame1)
        self.PredictButton.configure(textvariable = self.predictButtonLabel)
        self.PredictButton.configure(command=self.predictButtonPressed)
        self.PredictButton.configure(activebackground="#d9d9d9")
        self.PredictButton.configure(activeforeground="#000000")
        self.PredictButton.configure(background="#d9d9d9")
        self.PredictButton.configure(foreground="#000000")
        self.PredictButton.configure(highlightbackground="#d9d9d9")
        self.PredictButton.configure(highlightcolor="black")
        self.PredictButton.configure(relief=RAISED)
        # self.PredictButton.place(relx=0.61, rely=0.665, height=32, width=230)
        self.PredictButton.place(relx=0.205, rely=0.665, height=32, width=210)
        self.updatePredictButtonState()


        # CAM-Evolutions Button
        self.CAMButton = Button(self.Frame1)
        self.CAMButton.configure(text = "Build CAM-Evolutions")
        self.CAMButton.configure(command=self.camButtonPressed)
        self.CAMButton.configure(activebackground="#d9d9d9")
        self.CAMButton.configure(activeforeground="#000000")
        self.CAMButton.configure(background="#d9d9d9")
        self.CAMButton.configure(foreground="#000000")
        self.CAMButton.configure(highlightbackground="#d9d9d9")
        self.CAMButton.configure(highlightcolor="black")
        self.CAMButton.configure(relief=RAISED)
        self.CAMButton.place(relx=0.655, rely=0.665, height=32, width=200)


        # Images with their Labels
        self.Image1Label = Label(top)
        self.Image1Label.configure(activebackground="#f9f9f9")
        self.Image1Label.configure(activeforeground="black")
        self.Image1Label.configure(background="#d9d9d9")
        self.Image1Label.configure(foreground="#000000")
        self.Image1Label.configure(highlightbackground="#d9d9d9")
        self.Image1Label.configure(highlightcolor="black")
        self.Image1Label.configure(state=DISABLED)
        self.Image1Label.configure(text='''Your Image''')
        self.Image1Label.place(relx=0.04, rely=0.31, height=24, width=302)

        self.Image2Label = Label(top)
        self.Image2Label.configure(activebackground="#f9f9f9")
        self.Image2Label.configure(activeforeground="black")
        self.Image2Label.configure(background="#d9d9d9")
        self.Image2Label.configure(foreground="#000000")
        self.Image2Label.configure(highlightbackground="#d9d9d9")
        self.Image2Label.configure(highlightcolor="black")
        self.Image2Label.configure(state=DISABLED)
        self.Image2Label.configure(text='''Network Input''')
        self.Image2Label.place(relx=0.51, rely=0.31, height=24, width=302)

        self.Image1Canvas = Canvas(top)
        self.Image1Canvas.configure(background="#d9d9d9")
        self.Image1Canvas.configure(highlightbackground="#C9C9C9")
        self.Image1Canvas.configure(width=375)
        self.Image1Canvas.place(relx=0.04, rely=0.36, relheight=0.5, relwidth=0.44)

        self.Image2Canvas = Canvas(top)
        self.Image2Canvas.configure(background="#d9d9d9")
        self.Image2Canvas.configure(highlightbackground="#C9C9C9")
        self.Image2Canvas.configure(width=375)
        self.Image2Canvas.place(relx=0.51, rely=0.36, relheight=0.5, relwidth=0.44)


        # Benign/Malign Classification
        self.ClassificationCanvas = Canvas(top)
        self.ClassificationCanvas.configure(highlightbackground="#C9C9C9")
        self.ClassificationCanvas.configure(background="#d9d9d9")
        self.ClassificationCanvas.configure(width=475)
        self.ClassificationCanvas.place(relx=0.15, rely=0.91, relheight=0.06, relwidth=0.7)

        self.BenignLabel = Label(top)
        self.BenignLabel.configure(state=DISABLED)
        self.BenignLabel.configure(activebackground="#f9f9f9")
        self.BenignLabel.configure(activeforeground="black")
        self.BenignLabel.configure(background="#d9d9d9")
        self.BenignLabel.configure(foreground="#000000")
        self.BenignLabel.configure(highlightbackground="#d9d9d9")
        self.BenignLabel.configure(highlightcolor="black")
        self.BenignLabel.configure(borderwidth="0")
        self.BenignLabel.configure(relief="solid")
        self.BenignLabel.configure(text='''BENIGN''')
        self.BenignLabel.place(relx=0.04, rely=0.92, height=24, width=61)

        self.MalignLabel = Label(top)
        self.MalignLabel.configure(state=DISABLED)
        self.MalignLabel.configure(activebackground="#f9f9f9")
        self.MalignLabel.configure(activeforeground="black")
        self.MalignLabel.configure(background="#d9d9d9")
        self.MalignLabel.configure(foreground="#000000")
        self.MalignLabel.configure(highlightbackground="#d9d9d9")
        self.MalignLabel.configure(highlightcolor="black")
        self.MalignLabel.configure(relief="solid")
        self.MalignLabel.configure(borderwidth="0")
        self.MalignLabel.configure(text='''MALIGN''')
        self.MalignLabel.place(relx=0.87, rely=0.92, height=24, width=63)

        # Show Heatmaps on Label-Hover
        self.BenignLabel.bind("<Enter>", self.showBenignHeatmap)
        self.BenignLabel.bind("<Leave>", self.hideHeatmap)
        self.MalignLabel.bind("<Enter>", self.showMalignHeatmap)
        self.MalignLabel.bind("<Leave>", self.hideHeatmap)
        self.ClassificationCanvas.bind("<Enter>", self.showWeightedHeatmap)
        self.ClassificationCanvas.bind("<Leave>", self.hideHeatmap)



    def persistData(self):
        data = {
            'modelPath': self.modelPath.get(),
            'imagePath': self.imagePath.get(),
            'hairRemovalChecked': self.hairRemovalChecked.get(),
            'lesionSegmentChecked': self.lesionSegmentChecked.get(),
            'buildHeatmapsChecked': self.buildHeatmapsChecked.get()
        }

        with open('gui/GUI.pk', 'wb+') as fi:
            pickle.dump(data, fi)


    def setPath(self, path, pathLabel, pathShownLabel):
        pathLabel.set(path)
        pathShown = ".../" + io.get_file_with_parents(path, 2)
        pathShownLabel.set(pathShown)


    def loadData(self):
        if not os.path.exists('gui/GUI.pk'): return 

        with open('gui/GUI.pk', 'rb') as fi:
            data = pickle.load(fi)

        if 'modelPath' in data and os.path.exists(data['modelPath']):
            self.setPath(data['modelPath'], self.modelPath, self.modelPathShown)
        if 'imagePath' in data and os.path.exists(data['imagePath']):
            self.setPath(data['imagePath'], self.imagePath, self.imagePathShown)

        if 'hairRemovalChecked' in data:
            self.hairRemovalChecked.set(data['hairRemovalChecked'])
        if 'lesionSegmentChecked' in data:
            self.lesionSegmentChecked.set(data['lesionSegmentChecked'])
        if 'buildHeatmapsChecked' in data:
            self.buildHeatmapsChecked.set(data['buildHeatmapsChecked'])


    def selectModelPath(self):
        path = askopenfilename(title="Choose a model",filetypes=[("HDF5 Data Model",("*.h5","*.h5df"))])
        if path:
            self.setPath(path, self.modelPath, self.modelPathShown)
            self.updatePredictButtonState()
            self.persistData()


    def selectImagePath(self):
        path = askopenfilename(title="Choose an image",filetypes=[("Image",("*.jpg","*.jpeg","*.png"))])
        if path:
            self.setPath(path, self.imagePath, self.imagePathShown)
            self.updatePredictButtonState()
            self.persistData()
    

    def updatePredictButtonState(self):
        modelExists = self.modelPath.get() and os.path.exists(self.modelPath.get())
        imageExists = self.imagePath.get() and os.path.exists(self.imagePath.get())
        buttonState = NORMAL if (modelExists and imageExists) else DISABLED
        self.PredictButton.configure(state=buttonState)
    

    def predictButtonPressed(self):
        if not os.path.exists(self.imagePath.get()): return
        if not os.path.exists(self.modelPath.get()): return

        self.startLoading()

        # Load first image, preprocess second
        self.Image1Data = cv2.imread(self.imagePath.get())
        self.Image1Data = cv2.cvtColor(self.Image1Data, cv2.COLOR_BGR2RGB)
        self.showImage1()
        self.preprocessImage()
        self.showImage2()

        # Make Prediction (and Heatmaps if checked)
        if not hasattr(self, 'altMode') or not self.altMode:
            self.heatmaps = None
            buildHeatmaps = 'last' if self.buildHeatmapsChecked.get() == 1 else False
            self.predictions, self.prediction, _, self.heatmaps = predict.predict_image(self.modelPath.get(), 3, self.Image2Data, 250, buildHeatmaps)
            self.showPrediction()

        self.stopLoading()


    def camButtonPressed(self):
        output_dir = cam.save_class_activation_map_evolutions(self.modelPath.get(), 3, 2, self.imagePath.get(), self.Image2Data)
        call(["open", output_dir])


    def startLoading(self):
        self.isLoading = True
        self.top.config(cursor="wait")
        self.PredictButton.config(state=DISABLED)
        self.Image1Canvas.delete('all')
        self.Image2Canvas.delete('all')
        self.resetPrediction()


    def stopLoading(self):
        self.isLoading = False
        self.top.config(cursor="")
        self.PredictButton.config(state=NORMAL)


    def showImage1(self):
        if self.Image1Data is None: return
        self.Image1 = Image.fromarray(self.Image1Data.astype('uint8'))
        self.Image1Thumbnail = self.Image1.thumbnail((300,300), Image.ANTIALIAS)
        self.Image1PhotoImage = ImageTk.PhotoImage(self.Image1)
        self.Image1Canvas.create_image(150, 150, image=self.Image1PhotoImage, anchor=CENTER)

        height, width, _ = self.Image1Data.shape
        self.Image1Label.configure(text=f'Your Image ({width}x{height})')


    def showImage2(self):
        if self.Image2Data is None: return
        self.Image2 = Image.fromarray(self.Image2Data.astype('uint8'))
        self.Image2Thumbnail = self.Image2.thumbnail((300,300), Image.ANTIALIAS)
        self.Image2PhotoImage = ImageTk.PhotoImage(self.Image2)
        self.Image2Canvas.create_image(150, 150, image=self.Image2PhotoImage, anchor=CENTER)

        height, width, _ = self.Image2Data.shape
        self.Image2Label.configure(text=f'Network Input ({width}x{height})')


    def preprocessImage(self):
        if self.Image1Data is None: return

        image = self.Image1Data
        image = resize.apply(image.copy(), 250)

        # if self.hairRemovalChecked.get() == 1:
            # image = hair_removal.apply(image.copy())

        # if self.lesionSegmentChecked.get() == 1:
            # image = lesion_segmentation.apply(image.copy())

        self.Image2Data = image.copy()
        self.Image2DataBackup = image.copy()


    def resetPrediction(self):
        self.highlightLabel(self.MalignLabel, False)
        self.highlightLabel(self.BenignLabel, False)
        self.ClassificationCanvas.delete('all')


    def showPrediction(self):
        treshold = .5
        width = 475
        padding = 8

        # Highlight appropriate label
        self.highlightLabel(self.MalignLabel, self.prediction >= treshold)
        self.highlightLabel(self.BenignLabel, self.prediction < treshold)

        # Draw classification gradient and needle
        self.ClassificationCanvas.configure(highlightthickness="0")

        gradientImageSize = (width - 2 * padding, 24)
        gradientImage = colors.generate_gradient_image(*gradientImageSize)
        self.ClassificationImage = gradientImage
        self.ClassificationPhotoImage = ImageTk.PhotoImage(image=self.ClassificationImage, size=gradientImageSize)
        self.ClassificationCanvas.create_image(padding, 6, image=self.ClassificationPhotoImage, anchor=NW)

        self.NeedlePhotoImage = ImageTk.PhotoImage(file="gui/needle.gif")
        needleX = int(self.prediction*(width - 2*padding) + padding)
        self.ClassificationCanvas.create_image(needleX, 18, image=self.NeedlePhotoImage, anchor=CENTER)


    def highlightLabel(self, label, shouldHighlight):
        if shouldHighlight:
            label.configure(state=NORMAL)
            label.configure(borderwidth="2")
            label.configure(font=("TKDefaultFont", 13, "bold"))
        else:
            label.configure(state=DISABLED)
            label.configure(borderwidth="0")
            label.configure(font=("TKDefaultFont", 13))
    

    # Enable preprocesssing-Only by holding the Alt-Key (524320)
    def keyPress(self, e):
        if e.keycode == 524320: 
            self.predictButtonLabel.set("Preprocess only")
            self.altMode = True

    def keyRelease(self, e):
        if e.keycode == 524320: 
            self.predictButtonLabel.set("Preprocess & Predict")
            self.altMode = False
    

    # Show/Hide Activation Heatmaps
    def showBenignHeatmap(self, event):
        if self.isLoading: return
        if not hasattr(self, 'heatmaps'): return
        if not self.heatmaps: return
        self.hideHeatmap(None)
        self.Image2DataBackup = self.Image2Data
        self.Image2Data = cam.superimpose_heatmap_on_image(self.Image2Data, self.heatmaps[0])
        self.showImage2()

    def showMalignHeatmap(self, event):
        if self.isLoading: return
        if not hasattr(self, 'heatmaps'): return
        if not self.heatmaps: return
        self.hideHeatmap(None)
        self.Image2DataBackup = self.Image2Data
        self.Image2Data = cam.superimpose_heatmap_on_image(self.Image2Data, self.heatmaps[1])
        self.showImage2()

    def showWeightedHeatmap(self, event):
        if self.isLoading: return
        if not hasattr(self, 'heatmaps'): return
        if not self.heatmaps: return
        self.hideHeatmap(None)
        last_heatmaps = [self.heatmaps[0], self.heatmaps[1]]
        self.Image2DataBackup = self.Image2Data
        self.Image2Data = cam.superimpose_weighted_heatmap_on_image(self.Image2Data, last_heatmaps, self.predictions)
        self.showImage2()
    
    def hideHeatmap(self, event):
        if not hasattr(self, 'Image2DataBackup'): return
        self.Image2Data = self.Image2DataBackup
        self.showImage2()




if __name__ == "__main__":
    root = Tk()
    gui = PredictionGUI(root)
    root.bind('<KeyPress>', gui.keyPress)
    root.bind('<KeyRelease>', gui.keyRelease)
    root.resizable(False, False)

    # root.lift()
    # root.attributes('-topmost',True)
    # root.after_idle(root.attributes,'-topmost',False)

    # Bring window to front
    # Source: http://fyngyrz.com/?p=898&cpage=1
    os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')

    root.mainloop()



