import tkinter as tk
from tkinter import ttk
import numpy as np
import math
import os
from PIL import Image
from deblurgan.model import generator_model
from deblurgan.utils import preprocess_image, deprocess_image
from scripts.test import get_psnr, ssim
RESHAPE = (256, 256)

window = tk.Tk()

window.title('deblur')

var1 = tk.StringVar()
psnr = tk.Label(window, bg='pink',height=4, width=5, textvariable=var1)


img1 = None
img2 = None

def deblur(inputFile="input/0.png"):

    data = np.array( [preprocess_image(Image.open(inputFile)) ])
    g = generator_model()
    g.load_weights('generator.h5')
    generated_images = g.predict(x=data, batch_size=1)
    generated = np.array([deprocess_image(img) for img in generated_images])
    output = generated[0, :, :, :]
    im = Image.fromarray(output.astype(np.uint8))
    im.save('gui_output/results.png')

def start_deblur():
    
    global img2
    path = dropdown.get()

    try:
        deblur(path)

        sharp = Image.open("gui_output/results.png")
        sharp.save("gui_output/sharp.gif")
        sharp_gif = tk.PhotoImage(file="gui_output/sharp.gif")
        img2 = sharp_gif
        canvas1.create_image(250, 200 ,  image=img2)

        x_sharp = np.array(Image.open("gui_output/results.png").resize(RESHAPE))
        y_sharp = np.array(Image.open("gui_test/"+path).resize(RESHAPE))


        val_ssim = ssim(x_sharp, y_sharp)
        val_psnr = get_psnr(x_sharp, y_sharp)
        

        var1.set('psnr:\n'+
            str(round(val_psnr,4))+'\n'+
                'ssim:\n'+
                str(round(val_ssim,4))
            )

    except:
        tk.messagebox.showwarning("warning","something wrong")

def show_choice():
    global img1
    path = dropdown.get()
    blur = Image.open(path).resize(RESHAPE)
    blur.save("blur.gif")
    blur_gif = tk.PhotoImage(file="blur.gif")
    img1 = blur_gif
    canvas2.create_image(250, 200,  image=img1)


b1 = tk.Button(window, text='deblur', width=15, height=2, command=start_deblur)
b1.pack(side='bottom')
b2 = tk.Button(window, text='insert', width=15, height=2, command=show_choice)
b2.pack(side='bottom')


'''
var2 = tk.StringVar()
var2.set((11,22,33,44))
lb = tk.Listbox(window, listvariable=var2)
list_items = [1,2,3,4]
for item in list_items:
    lb.insert('end', item)
lb.insert(1, 'first')
lb.insert(2, 'second')
lb.delete(2)
lb.pack(side='right')
'''


#os.listdir()
dropdown = ttk.Combobox(
                window,
        )

catalog = os.listdir()
print(catalog)
i=0
while i < len(catalog):
    if 'gif' not in catalog[i] and 'png' not in catalog[i]:
        catalog[i]='toBeDeleted'
    i = i+1
    
while catalog.count('toBeDeleted') != 0:
    catalog.remove('toBeDeleted')    
    
print (catalog)

dropdown['values'] = catalog
dropdown.pack(side = 'top', anchor = 'n',
                    fill = 'x', expand = 1, padx = 8, pady = 8)

canvas1 = tk.Canvas(window, width=500, height=400,bg='white')
canvas1.pack(side='right')
psnr.pack(side='right')
canvas2 = tk.Canvas(window, width=500, height=400,bg='white')
canvas2.pack(side='right')




window.mainloop()





