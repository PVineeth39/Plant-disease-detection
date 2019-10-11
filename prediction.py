# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:56:32 2019

@author: pothi
"""
import numpy as np
from keras.preprocessing import image

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import messagebox
import os
from keras.models import load_model
classifier = load_model('mymodel.h5')
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img():
    x = openfn()
    test_image = image.load_img(x, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict_classes(test_image)
    print(result)
    index=['Pepper bell Bacterial spot','Pepper bell healthy','Potato Early blight','Potato Late blight','Potato healthy','Tomato Bacterial spot','Tomato Early blight','Tomato Late blight','Tomato Leaf Mold','Tomato Septoria leaf spot','Tomato Spider mites Two spotted spider mite','Tomato Target Spot','Tomato YellowLeaf Curl Virus','Tomato Tomato mosaic virus','Tomato healthy']
    #index=['Pepper bell Bacterial spot','Pepper bell healthy','Potato Early blight','Potato healthy','Potato Late blight','Tomato Target Spot','Tomato mosiac virus','Tomato Yellow Leaf CurlVirus','Tomato Bacterial spot','Tomato Early blight','Tomato healthy','Tomato Late blight','Tomato leaf mold','Tomato Septoria leaf spot','Tomato Spider mites']
    label = Label( root, text="Prediction : "+index[result[0]-1])
    res=messagebox.showinfo('Results',index[result[0]-1])
    label.pack()
    img = Image.open(x)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()
    
root.configure(background='light green')
btn = Button(root, text='Upload image',background='black',fg='yellow' ,command=open_img).pack()

root.mainloop()