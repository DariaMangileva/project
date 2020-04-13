import os
import gc
import datetime
import numpy as np
import pandas as pd
import cv2
import time
from copy import deepcopy
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras import backend as K
from keras.utils import Sequence
import matplotlib 
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter
#from IPython.display import display
from keras_tqdm import TQDMCallback


# Change to root path
if os.path.basename(os.getcwd()) != 'PConv-Keras':
    os.chdir('..')

from pconv_model import PConvUnet
#from libs.util import MaskGenerator

# %load_ext autoreload
# %autoreload 2
plt.ioff()

# SETTINGS
TRAIN_DIR_ori = r"/misc/home/u2592/TRAIN2/ori"
TRAIN_DIR_mask = r"/misc/home/u2592/TRAIN2/mask"
VAL_DIR_ori = r"/misc/home/u2592/VAL2/ori"
VAL_DIR_mask = r"/misc/home/u2592/VAL2/mask"
TEST_DIR_ori = r"/misc/home/u2592/TEST2/ori"
TEST_DIR_mask = r"/misc/home/u2592/TEST2/mask"
BATCH_SIZE = 4

"""# Network Training
Having implemented and tested all the components of the final networks in steps 1-3, we are now ready to train the network on a large dataset (ImageNet).

Creating train & test data generator
# """

class AugmentingDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory1, directory2, *args, **kwargs):
        generator1 = super().flow_from_directory(directory1, class_mode=None, *args, **kwargs)
        generator2 = super().flow_from_directory(directory2, class_mode=None, *args, **kwargs)        
        seed = None if 'seed' not in kwargs else kwargs['seed']
        while True:
            
            # Get augmentend image samples
            ori = next(generator1)
                        
            

            # Get masks for each image sample            
            mask = next(generator2)
          	



            masked = deepcopy(ori)
            masked[mask==0] = 1
            
            

            
            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori
            

# Create training generator
train_datagen = AugmentingDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR_ori, 
    TRAIN_DIR_mask,
    target_size=(512, 512), 
    batch_size=BATCH_SIZE,
    color_mode = 'rgb'
)

# Create validation generator
val_datagen = AugmentingDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR_ori, 
    VAL_DIR_mask, 
    target_size=(512, 512), 
    batch_size=BATCH_SIZE,
    color_mode = 'rgb',  
    seed=42
)

# Create testing generator
test_datagen = AugmentingDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR_ori, 
    TEST_DIR_mask, 
    target_size=(512, 512), 
    batch_size=BATCH_SIZE,
    color_mode = 'rgb', 
    seed=42
)

# Pick out an example
test_data = next(test_generator)
(masked, mask), ori = test_data

# Show side by side
# for i in range(len(ori)):
	
#     _, axes = plt.subplots(1, 3, figsize=(20, 5))
#     axes[0].imshow(masked[i,:,:,:])
#     axes[1].imshow(mask[i,:,:,:] * 1.)
#     axes[2].imshow(ori[i,:,:,:])
#     plt.show()
	

"""# Training on ImageNet"""

def plot_callback(model):
    """Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk"""
    
    # Get samples & Display them        
    pred_img = model.predict([masked, mask])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Clear current output and display test images
    for i in range(len(ori)):
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(masked[i,:,:,:], cmap = 'gray')
        axes[1].imshow(pred_img[i,:,:,:], cmap = 'gray')
        axes[2].imshow(ori[i,:,:,:], cmap = 'gray')
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[2].set_title('Original Image')             
        plt.savefig(r'/misc/home/u2592/image/img_{i}_{pred_time}.png'.format(i = i, pred_time = pred_time))
        plt.close()

"""## Phase 1 - with batch normalization"""

model = PConvUnet(vgg_weights='/misc/home/u2592/data/pytorch_to_keras_vgg16.h5')
model.load('/misc/home/u2592/data/phase2/weights.20-0.07.h5', train_bn=False, lr=0.00005)
FOLDER = r'/misc/home/u2592/data/phase2'
# Run training for certain amount of epochs
model.fit_generator(
    train_generator, 
    steps_per_epoch=3522,
    validation_data=val_generator,
    validation_steps=499,
    epochs=20,  
    verbose=0,
    callbacks=[
        TensorBoard(
            log_dir=FOLDER,
            write_graph=False
        ),
        ModelCheckpoint(
            '/misc/home/u2592/data/phase2/weights.{epoch:02d}-{loss:.2f}.h5',
            monitor='val_loss', 
            save_best_only=True, 
            save_weights_only=True),
        TQDMCallback()
    ]
)
