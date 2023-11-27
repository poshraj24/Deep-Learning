import numpy as np
import matplotlib.pyplot as plt

class Checker():
    def __init__(self, resolution:int, tile_size:int):
        assert (resolution%(2*tile_size)==0)
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None
    
   
    def draw(self):
        rows = self.resolution // self.tile_size
        colls = self.resolution // self.tile_size
        mask = (np.arange(rows)[:, None] + np.arange(colls)) % 2
        tile = np.ones((self.tile_size, self.tile_size), dtype=int)
        tiled_mask = np.kron(mask, tile)
        self.output = tiled_mask
        return tiled_mask.copy()
    
    def show(self):
        if self.output is None:
            self.draw()
        print(self.output)
        plt.figure(figsize=(8,8))
        plt.imshow(self.output, cmap='gray')
        plt.title("Output")
        plt.show()
           
#resolution = 8
#tile_size = 2

#checkerboard = Checker(resolution, tile_size)
#checkerboard.show()
class Circle():
    def __init__(self, resolution:int, radius:int, position:tuple):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output=None

  
    def draw(self):
        a = np.arange(self.resolution)
        b = np.arange(self.resolution)
        aa, bb = np.meshgrid(a, b)
        binary_circle = (aa - self.position[0])**2 + (bb - self.position[1])**2 <= self.radius**2
        self.output = np.where(binary_circle, 1, 0)
        return self.output.copy()

    def show(self):
        plt.figure(figsize=(10,10))
        plt.imshow(self.output, cmap='gray')
        plt.title("Output")
        plt.show()
      
class Spectrum():
    def __init__(self, resolution:int):
        self.resolution=resolution
        self.output=None

    def draw(self):
        spectrum = np.zeros((self.resolution,self.resolution,3))
        spectrum[:,:,0]=np.linspace(0,1,self.resolution)
        spectrum[:,:,1]=np.linspace(0,1,self.resolution).reshape(-1,1)
        spectrum[:,:,2]=np.linspace(1,0,self.resolution)

        self.output = spectrum
        return spectrum.copy()
             
      
    def show(self):
        plt.imshow(self.output)
        plt.show()



 