from tkinter import *
from tkinter.filedialog import Open
import cv2
from keras.models import model_from_json
import numpy as np
from PIL import Image, ImageTk

model_architecture = "Rice_config.json"
model_weights = "Rice-weights-151-0.9900.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
x_label = ['Arborio','Barley','Basmati','Brown_rice','Ipsala','Japonica',
           'Jasmine','Jungwon','Karacadag','Sticky_rice']
class Main(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
  
    def initUI(self):
        self.parent.title("Rice Classification")
        self.pack(fill=BOTH, expand=1)
  
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)
  
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open Picture", command=self.onOpenPicture)
        fileMenu.add_command(label="Open Camera", command=self.onOpenCam)
        fileMenu.add_command(label="Open Video", command=self.onOpenVideo)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=fileMenu)
  
    def onOpenPicture(self):
        global ftypes
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
  
        if fl != '':
            global img
            global imgin
            imgin = cv2.imread(fl,cv2.IMREAD_GRAYSCALE)
            imgin = cv2.resize(imgin[600:600+750, 600:600+750],dsize=(250,250))
            threshold = 150
            ret, img = cv2.threshold(imgin, threshold,255,cv2.THRESH_BINARY)
            img = cv2.medianBlur(img,5)
            img = cv2.medianBlur(img,5)
            w = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, w)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, w)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, w)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, w)
            count, label = cv2.connectedComponents(img)
            M, N = label.shape
            for x in range(0, M):
                for y in range(0, N):
                    r = label[x, y]
                    if (r == 0):
                        imgin[x,y] = 0
            
            img = imgin.reshape(1,250,250,1)
            img = img.astype('float32')
            img /=255.0
            sample = np.array(img)
            predictions = np.argmax(model.predict(sample), axis=-1)
            print("predictions:", predictions)
            print("predictions label:", x_label[int(predictions)])
            cv2.putText(imgin,x_label[int(predictions)],(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
            cv2.namedWindow(fl, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(fl, imgin)
    def onOpenCam(self):
        global imgin
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            if not ret:
                continue
            M,N = frame.shape[:2]
            a = int((M-N)/2)
            if a < 0:
                a = 0
            b = int((N-M)/2)
            if b < 0:
                b = 0
            imgin = frame[a:a + min(M,N),b:b+ min(M,N)]
            new_img = cv2.resize(imgin,dsize=(480,480))
            # print(imgin.shape)
            imgin = cv2.cvtColor(imgin,cv2.COLOR_RGB2GRAY)
            imgin = cv2.resize(imgin[0:0+500, 300:300+500],dsize=(250,250))
            threshold = 150
            ret, img = cv2.threshold(imgin, threshold,255,cv2.THRESH_BINARY)
            img = cv2.medianBlur(img,5)
            img = cv2.medianBlur(img,5)
            w = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, w)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, w)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, w)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, w)
            count, label = cv2.connectedComponents(img)
            M, N = label.shape
            for x in range(0, M):
                for y in range(0, N):
                    r = label[x, y]
                    if (r == 0):
                        imgin[x,y] = 0
            img = imgin.reshape(1,250,250,1)
            img = img.astype('float32')
            img /=255.0
            sample = np.array(img)
            predictions = np.argmax(model.predict(sample), axis=-1)
            print("predictions:", predictions)
            print("predictions label:", x_label[int(predictions)])
            cv2.putText(new_img,x_label[int(predictions)],(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
            cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Camera", new_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
    def onOpenVideo(self):
        global ftypes
        ftypes = [('Videos', '*.mp4 *.avi *.3gp *.wmv')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
  
        if fl != '':
            global imgin
            cap = cv2.VideoCapture(fl)
            while(True):
                ret, frame = cap.read()
                if not ret:
                    continue
                M,N = frame.shape[:2]
                a = int((M-N)/2)
                if a < 0:
                    a = 0
                b = int((N-M)/2)
                if b < 0:
                    b = 0
                imgin = frame[a:a + min(M,N),b:b+ min(M,N)]
                new_img = cv2.resize(imgin,dsize=(480,480))
                # print(imgin.shape)
                imgin = cv2.cvtColor(imgin,cv2.COLOR_RGB2GRAY)
                # imgin = cv2.resize(imgin[0:0+500, 300:300+500],dsize=(500,500))
                imgin = cv2.resize(imgin,dsize=(250,250))
                threshold = 150
                ret, img = cv2.threshold(imgin, threshold,255,cv2.THRESH_BINARY)
                img = cv2.medianBlur(img,5)
                img = cv2.medianBlur(img,5)
                w = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, w)
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, w)
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, w)
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, w)
                count, label = cv2.connectedComponents(img)
                M, N = label.shape
                for x in range(0, M):
                    for y in range(0, N):
                        r = label[x, y]
                        if (r == 0):
                            imgin[x,y] = 0
                img = imgin.reshape(1,250,250,1)
                img = img.astype('float32')
                img /=255.0
                sample = np.array(img)
                predictions = np.argmax(model.predict(sample), axis=-1)
                print("predictions:", predictions)
                print("predictions label:", x_label[int(predictions)])
                cv2.putText(new_img,x_label[int(predictions)],(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
                cv2.namedWindow(fl, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(fl, new_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break

root = Tk()
Main(root)
root.geometry("1000x709")
canvas = Canvas(root,width=999,height=708)
canvas.pack()
pilImage = Image.open("background.png")
pilImage = pilImage.resize((1000, 709), Image.ANTIALIAS)
image = ImageTk.PhotoImage(pilImage)
imagesprite = canvas.create_image(500,355,image=image)
root.mainloop()
