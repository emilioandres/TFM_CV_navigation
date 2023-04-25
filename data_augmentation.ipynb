from keras.preprocessing.image import ImageDataGenerator  
from keras.applications import densenet  
from keras.models import Sequential, Model, load_model  
from keras.layers import Conv2D, MaxPooling2D  
from keras.layers import Activation, Dropout, Flatten, Dense  
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback  
from keras import regularizers  
from keras import backend as K  
K.set_learning_phase(1)

img_width, img_height = 128,128 
nb_train_samples = 1449  
nb_validation_samples = 8041  
epochs = 10  
batch_size = 256   



datagen = ImageDataGenerator(  
    rescale=1,
    zoom_range=0.2,
    rotation_range = 5,
    horizontal_flip=True)



segm_generator = datagen.flow(
    X_mask,
    y=None,
    batch_size=batch_size,
    shuffle=True,
    sample_weight=None,
    seed=42,
    ignore_class_split=False,
    subset=None
)

rgb_generator = datagen.flow(
    X_rgb_re,
    y=None,
    batch_size=batch_size,
    shuffle=True,
    sample_weight=None,
    seed=42,
    ignore_class_split=False,
    subset=None
)
depth_generator = datagen.flow(
    X_depth_re,
    y=None,
    batch_size=batch_size,
    shuffle=True,
    sample_weight=None,
    seed=42,
    ignore_class_split=False,
    subset=None
)
