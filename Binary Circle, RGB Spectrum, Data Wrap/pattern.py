import numpy as np
import matplotlib.pyplot as plt

class Checker():
    def __init__(self, resolution: int, tile_size: int) -> None:
        
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((resolution, resolution))
        

            
    def draw(self):
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        #create a coordinate grid of x and y values.
        #get the alternating 0 and 1 values for the checkerboard pattern
        xv, yv = np.meshgrid(x // self.tile_size, y // self.tile_size, sparse=False) 
        self.output = (xv % 2 != yv % 2).astype(int) # boolean matrix where True is 1 and False is 0
        return self.output.copy()    
    
    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()
        
    

class Circle():
    def __init__(self, resolution: int, radius:int, position:tuple) -> None:
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output=None
        

    def draw(self):
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        xv, yv = np.meshgrid(x, y, sparse=False)
        self.output = (np.sqrt((xv - self.position[0])**2 + (yv - self.position[1])**2) < self.radius).astype(int)
        return self.output.copy()

    
    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Spectrum():
    def __init__(self,resolution:int) -> None:
        self.resolution = resolution
        self.output = None

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution, 3))
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        xv, yv = np.meshgrid(x, y, sparse=False)
        
        self.output[:, :, 0] = xv
        self.output[:, :, 1] = yv
        self.output[:, :, 2] = np.linspace(1,0,self.resolution)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.show()

