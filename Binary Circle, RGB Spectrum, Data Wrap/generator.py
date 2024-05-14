import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.misc
import os.path
from skimage.transform import resize, rotate

class ImageGenerator():
    def __init__ (self, file_path:str, label_path:str, batch_size:int, image_size, rotation=False, mirroring=False, shuffle=False):
        self.file_path=file_path
        self.label_path=label_path
        self.batch_size=batch_size
        self.image_size=image_size
        self.rotation=rotation
        self.mirroring=mirroring
        self.shuffle=shuffle
        self.dictionary = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9:'truck'}
        #to read the content and parse json value to python dictionary:
        with open(self.label_path, 'r+') as fp:
            self.labels = json.load(fp)
        #initializing epochs and batches
        self.current_epoch_count = -1
        self.current_batches = {self.current_epoch_count:[]} #current batches are stored as dictionary
            
    def next(self):
        # checks if the current epoch's batches are already available
        if not self.current_batches.get(self.current_epoch_count):
            #initializes the batches
            #self.current_epoch_count += 1
            files = os.listdir(self.file_path)
            nfiles = len(files)
            
            #shuffles the files if required
            if self.shuffle:
                np.random.shuffle(files)
            self.current_epoch_count += 1
            #creates batches of files based on the batch size.
            rem = nfiles % self.batch_size
            if rem > 0:
                files += files[:self.batch_size - rem]
            
            batches = [files[i: i + self.batch_size] for i in range(0, len(files), self.batch_size)]
            self.current_batches[self.current_epoch_count] = batches

        #pops the first batch of files from the current epoch's batches
        batch_files = self.current_batches[self.current_epoch_count].pop(0)
        
        #initializes arrays for images and labels.
        images = np.zeros((self.batch_size, *self.image_size))
        labels = np.zeros((self.batch_size, 1), dtype=int)
        
        #iterate over the batch of files, load each image, retrieve the corresponding label
        for i, file in enumerate(batch_files):
            img = np.load(os.path.join(self.file_path, file))
            label = int(self.labels[file.split(".")[0]])

            #include resizing method 
            img = resize(img, self.image_size)
            img = self.manipulation(img)

            images[i] = img
            labels[i] = label
        
        return images, labels

    #Implementing the functionalities- mirroing and rotation, shuffling is done above
    def manipulation(self, img):
        if self.mirroring and np.random.randint(2) % 2 == 0:
            img = np.fliplr(img) #horizontal flipping of image/array
        if self.rotation:
            angles = [90, 180, 270]
            img = rotate(img, np.random.choice(angles))
        return img
    
     # returns the current epoch number
    def current_epoch(self):
        return self.current_epoch_count
    
    #returns the class name for a specific input
    def class_name(self, a):
        return self.dictionary[a]
    
    # generate a batch using next() and plot it
    def show(self):
        fig = plt.figure()
        axes = fig.subplots(2, 5)
        images, labels = self.next()
                
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i])
            ax.set_title(str(self.class_name(labels[i][0])))
         
        plt.show()   
        
