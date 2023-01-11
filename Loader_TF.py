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
    def __init__(self,*,path,cpu,image_size=(256,256),batch_size) :
        
        self.Cpu = cpu
        
        Category = sorted([ i.split("/")[-1] for i in glob(path + "/*") ]) 
        self.Category =  {Category[i] : i for i in range(len(Category))}
        FileName = glob(path + "/*/*")
        np.random.shuffle(FileName)
        
        self.FileName = tf.constant(FileName)
        self.Label    = tf.constant([self.Category[i.split("/")[-2]] for i in FileName])
        
        self.Total = self.FileName.shape[0]
        self.Saved = 0
        self.Batch_Size = batch_size
        self.Image_Size = image_size
    
    def TFM(self, path, label):
        image_string = tf.io.read_file(path)           
        image_decoded = tf.image.decode_jpeg(image_string)  
        image_resized = tf.image.resize(image_decoded, self.Image_Size) / 255.0
        
        label = self.Category[label]
        return image_resized , label

    def Get(self,begin = None,end = None):
        if begin == None : begin = 0
        if end   == None : end = self.Total 
        
        self.Saved = (end - begin)
        
        self.Dataset = tf.data.Dataset.from_tensor_slices((self.FileName[begin:end],self.Label[begin:end]))
        self.Dataset = self.Dataset.map(map_func = self.TFM , 
                                        num_parallel_calls = self.Cpu)
        
        self.Dataset = self.Dataset.shuffle(buffer_size=self.Saved)    
        self.Dataset = self.Dataset.batch(batch_size = self.Batch_Size)
        self.Dataset = self.Dataset.prefetch(self.Cpu)
        
        
        
class My_Loader(Base_Loader) :
    def TFM(self, path,label):
        image_string = tf.io.read_file(path)           
        image_decoded = tf.image.decode_jpeg(image_string)  
        image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
        return image_resized ,label
