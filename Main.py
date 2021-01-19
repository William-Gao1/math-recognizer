import Recognition
from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import ImageGrab
from threading import Timer
from PIL import Image
import numpy
import cv2


class Paint(object):

    
    WIDTH = 600
    HEIGHT = 600

    def __init__(self):
        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.timer = Timer(1, self.read)
        self.c = Canvas(self.root, bg='white', width=self.WIDTH, height=self.HEIGHT)
        self.c.grid(row=1, columnspan=5)
        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 7
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        

    def read(self):
        print("hi")
        print(Recognition.readNumber(self.getter()))

    def getter(self):
        # # save postscipt image 
        # self.c.postscript(file = 'image' + '.eps') 
        # # use PIL to convert to PNG 
        # original = Image.open('image.eps')
        # original. save("converted.png", format="png")
        # converted = Image. open("converted.png")
        # return numpy.array(converted)
        
        print('\n def _snapCanvas(self):')
        canvas = self.getBBox() # Get Window Coordinates of Canvas
        grabcanvas = ImageGrab.grab(bbox=canvas).save('img.png')
        #grabcanvas.show()
        #return cv2.imread('img.png', cv2.IMREAD_COLOR)
        return 'img.png'
        
        
        #img.save(fileName + '.png', 'png')

    def getBBox(self):
        
        x=self.c.winfo_rootx()+self.c.winfo_x()
        y=self.c.winfo_rooty()+self.c.winfo_y()+100
        x1=x+1200
        y1=y+1100
        box=(x+50,y,x1,y1)
        return box
        

    def restartTimer(self):
        self.timer = Timer(1, self.read)
        self.timer.start()

    def close(self):
        self.timer.cancel()
        self.root.destroy()

    def paint(self, event):
        if self.timer.is_alive():
            self.timer.cancel
        paint_color = 'black'
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=1)
            
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None
        self.restartTimer()


if __name__ == '__main__':
    Paint()