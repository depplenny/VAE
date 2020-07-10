import numpy as np
import cv2
import os
import time
import h5py
import sys
import glob
import matplotlib.pyplot as plt

class TrainingDatasetLoader(object):
    def __init__(self, data_path):

        print ("Opening {}".format(data_path))
        sys.stdout.flush()
        
        self.cache = h5py.File(data_path, 'r')
        
        print ("Loading data into memory...")
        sys.stdout.flush()
        #cache contains 
        self.images = self.cache['images'][:]
        self.labels = self.cache['labels'][:].astype(np.float32)
        self.n_train_samples = self.images.shape[0]
        self.train_inds = np.random.permutation(np.arange(self.n_train_samples))
        self.face_inds = self.train_inds[ self.labels[self.train_inds, 0] == 1.0 ]
        self.non_face_inds = self.train_inds[ self.labels[self.train_inds, 0] != 1.0 ]

    def get_dataset_size(self):
        return self.n_train_samples
        
    def get_all_faces(self):
        return self.images[ self.face_inds ]
        
    def get_all_non_faces(self):
        return self.images[ self.non_face_inds ]

    def get_batch(self, n, only_faces=False, return_inds=False):
        # Randomness required by SDG
        if only_faces:
            selected_inds = np.random.choice(self.face_inds, size=n, replace=False)
        else:
            selected_face_inds = np.random.choice(self.face_inds, size=n//2, replace=False)
            selected_non_face_inds = np.random.choice(self.non_face_inds, size=n//2, replace=False)
            selected_inds = np.concatenate((selected_face_inds, selected_non_face_inds))

        sorted_inds = np.sort(selected_inds)
        train_img = (self.images[sorted_inds,:,:,::-1]/255.).astype(np.float32)
        train_label = self.labels[sorted_inds,...]
        return (train_img, train_label, sorted_inds) if return_inds else (train_img, train_label)

    
def plot_sample(x,vae,idx=0):
    plt.figure(figsize=(3,3))
    plt.subplot(1, 2, 1)

    plt.imshow(x[idx])
    plt.grid(False)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    _, _, _, recon = vae(x)
    recon = np.clip(recon, 0, 1)
    plt.imshow(recon[idx])
    plt.grid(False)
    plt.axis('off')

    plt.show()
    
