from glob import glob
from time import time
from multiprocessing.dummy import Pool as ThreadPool
import PIL
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
class Timer :
    clock = time()
    clock_work =  time()
    clock_base =  time()
    def Pass(active = False):
        
        if active : print("Passed Time : %.2f Sec " % (time() - Timer.clock) )
        Timer.clock= time()
       
    def Active(period = 1,refresh = True) :
        if  time() - Timer.clock_work  > period : 
            if refresh : Timer.clock_work = time()
            return True
        else : 
            return False
    
    def Counter(refresh = False):
        if refresh : Timer.clock_base = time()
        return time() - Timer.clock_base
    
class Base_Loader:
    def __init__(self,*,path,cpu,image_size=(256,256)) :
        
        self.Cpu = cpu
        self.Category = np.array( sorted([ i.split("/")[-1] for i in glob(path + "/*") ]) )
        self.File = np.array(glob(path + "/*/*"))
        np.random.shuffle(self.File)
        self.Total = len(self.File)
        self.Saved = 0
        
        self.Image = np.array([])
        self.Label = np.array([])
        self.Name  = np.array([])
        
        self.Image_Size = image_size

        self.Progress = 0
        self.Current = 0
        
        self.isTensor = False
    def TFM_Wrap(self,TFM):
        def TFM_Pipline(data):
            number = data[0]
            path = data[1]
            self.Current += 1
            if Timer.Active(0.01) or self.Current == self.Progress: 
                print('\r' + '[Progress]:[%s%s]%.2f%%;Passed Time %.2f Sec' % (
                '█' * int((self.Current)*20/self.Progress), ' ' * (20-int((self.Current)*20/self.Progress)),
                float((self.Current)/self.Progress*100),Timer.Counter()), end='')
                    
            return TFM(path)
        return TFM_Pipline
    def TFM(self,path):
        
        img = PIL.Image.open(path)
        img   = np.array( img.resize(self.Image_Size) )
        return img
    
    def Validation_F1(self,model , chunk = 50):
        TP = { i:0 for i in self.Category.keys() }
        TN = { i:0 for i in self.Category.keys() }
        FP = { i:0 for i in self.Category.keys() }
        FN = { i:0 for i in self.Category.keys() }
        EX = { i:0 for i in self.Category.keys() }
        
        Total_Chunk = math.ceil(self.Saved / chunk)
        predict_label = []
        

        for i in tqdm(range(Total_Chunk)):
            tail,head = chunk*i , chunk*(i+1) 
            if i == Total_Chunk-1 : predict_label.append(model(self.Image[tail:]))
            else :  predict_label.append(model(self.Image[tail:head]))
        predict_label = np.hstack(predict_label)
    
        for i in range(self.Saved) :
            correct_label = self.Label[i] 
            EX[ correct_label ] += 1
            if correct_label == predict_label[i] : TP[ correct_label ] += 1
            else : 
                FP[ correct_label ] += 1
                FN[ predict_label[i] ] += 1
                
        ACC = {i : round((TP[i] + TN[i])*100 / (TP[i]+TN[i]+FP[i]+FN[i]),3) \
               if TP[i]+TN[i]+FP[i]+FN[i] != 0 else 0 for i in self.Category.keys() }
        PRE = {i : round((TP[i] )*100 / (TP[i]+FP[i]),3) \
               if (TP[i]+FP[i]) != 0 else 0 for i in self.Category.keys() }
        REC = {i : round((TP[i] )*100 / (TP[i]+FN[i]),3) \
               if (TP[i]+FN[i]) != 0 else 0 for i in self.Category.keys() }
        F1  = {i : round((2*PRE[i]*REC[i]) / (PRE[i] + REC[i]),3) \
               if (PRE[i] + REC[i]) != 0 else 0 for i in self.Category.keys() }
        RES = {i : [EX[i],TP[i],TN[i],FP[i],FN[i],ACC[i],PRE[i],REC[i],F1[i]] for i in self.Category.keys()}
        
        F1_Score = pd.DataFrame.from_dict(RES, orient='index',columns=["EX",'TP', 'TN', 'FP', 'FN','ACC','PRE','REC','F1'])
        print(F1_Score)    
        
        x = [i for i in range(1,1+len(self.Category))] 
        h = [F1[i] for i in self.Category.keys()]
        label = [i for i in self.Category.keys()]
        plt.barh(x,h,tick_label=label,height=0.5)  
        plt.show()    
    def Show(self,begin,end,shape ):
        
        h = shape[1]
        w = shape[0]
        
        
        fig, axes = plt.subplots(h, w,figsize=(w*4, h*4))
        
        if h == 1 : axes = np.expand_dims(axes,axis = 0)
        if w == 1 : axes = np.expand_dims(axes,axis = 1)
            
        size = h*w
        for i in range(size):
            axes[(i)//w,(i)%w].imshow(self.Image[begin + i])
            axes[(i)//w,(i)%w].set_title(self.Label[begin + i])
        plt.show()    
        
    def Info(self):
        print("Total %d Data Saved in Channel"%(self.Total))
        print("Now %d Data Saving in Memory"%(self.Saved))
    
    def Get(self,begin = None,end = None, clear = False):
        if begin == None : begin = 0
        if end   == None : end = self.Total 
            
        Timer.Counter(True)
        TFM = self.TFM_Wrap(self.TFM)
        self.Progress = end - begin
        self.Current = 0
        with ThreadPool(self.Cpu) as T:
                Image = np.array(T.map(TFM , enumerate(self.File[begin:end])),dtype = np.uint8)
                
        print(" ")
        Label = np.array( [path.split("/")[-2] for path in self.File[begin:end]])
        Name  = np.array( [path.split("/")[-1] for path in self.File[begin:end]])
        
        if clear or self.Saved == 0: self.Image,self.Label,self.Name = Image,Label,Name
        else : self.Image,self.Label,self.Name  = np.append(self.Image,Image,axis = 0),np.append(self.Label,Label),np.append(self.Name,Name)
            
        self.Saved += (end - begin)
    def Clear(self,begin = None,end = None):
        if begin == None : begin = 0
        if end   == None : end = self.Saved
        self.Image = np.append( self.Image[:begin] , self.Image[end:] ,axis = 0)
        self.Label = np.append( self.Label[:begin] , self.Label[end:] )
        self.Name = np.append( self.Name[:begin] , self.Name[end:] )
        self.Saved -= (end - begin)
        
    def Copy(self,copied,begin = None,end = None):
        if not begin : begin = 0
        if not end   : end   = copied.Saved 
        
        self.Image = copied.Image[begin:end]
        self.Label = copied.Label[begin:end]
        self.Name  = copied.Name[begin:end]
        self.Saved = min(copied.Saved - begin,end - begin) 
    
    def Packet(self) :
        self.isTensor = True
        self.Dataset = tf.data.Dataset.from_tensor_slices((obj.Image, obj.Label))
    
class My_Loader(Base_Loader) :
    def TFM(self,path):
        img = PIL.Image.open(path)
        img   = np.array( img.resize(self.Image_Size) )
        return img