from sklearn.model_selection import KFold
# Definir el número de divisiones y el modelo
num_folds = 3
kfold = KFold(n_splits=num_folds, shuffle=True)
kfold.get_n_splits(dataset_img)
print(kfold)

print('Img dataset ',dataset_img.shape)

print('Depth dataset ',depth_result.shape)

print('Y dataset ',X_mask.shape)

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Dropout, Reshape, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow as tf

def RedNet(img_height, img_width, nclasses,nfilters,size_stride,size_kernel,dilation):
    input1 = Input(shape=(img_height, img_width, 3)) #rgb
    input2 = Input(shape=(img_height, img_width, 3)) #depth
    #depth
    x1 = Conv2D(nfilters*1, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(input2)
    x1 = Conv2D(nfilters*1, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x1)


    #rgb
    x2 = Conv2D(nfilters*1, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(input1)
    x2 = Conv2D(nfilters*1, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x2)
    x3 = Conv2D(nfilters*1, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x2)
    x3 = Conv2D(nfilters*1, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x3)
    x3 = MaxPooling2D((2, 2), padding="same")(x3)

    #rgb depth fusion
    x = Add()([x1, x2])
    x = Conv2D(nfilters*1, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
    x = Conv2D(nfilters*1, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    #rgb
    x4 = Conv2D(nfilters*2, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x3)
    x4 = Conv2D(nfilters*2, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x4)
    x5 = Conv2D(nfilters*2, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x4)
    x5 = Conv2D(nfilters*2, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x5)
    x5 = MaxPooling2D((2, 2), padding="same")(x5)

    #depth
    x6 = Conv2D(nfilters*2, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
    x6 = Conv2D(nfilters*2, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x6)

    #rgb depth fusion
    x = Add()([x4, x6])
    x = Conv2D(nfilters*2, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
    x = Conv2D(nfilters*2, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    #rgb
    x7 = Conv2D(nfilters*4, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x5)
    x7 = Conv2D(nfilters*4, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x7)
    x8 = Conv2D(nfilters*4, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x7)
    x8 = Conv2D(nfilters*4, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x8)
    x8 = MaxPooling2D((2, 2), padding="same")(x8)

    #depth
    x9 = Conv2D(nfilters*4, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
    x9 = Conv2D(nfilters*4, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x9)


    #rgb depth fusion
    x = Add()([x7, x9])
    x = Conv2D(nfilters*4, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
    x = Conv2D(nfilters*4, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    #rgb
    x10 = Conv2D(nfilters*8, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x8)
    x10 = Conv2D(nfilters*8, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x10)
    x11 = Conv2D(nfilters*8, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x10)
    x11 = Conv2D(nfilters*8, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x11)
    x11 = MaxPooling2D((2, 2), padding="same")(x11)

    #depth
    x12 = Conv2D(nfilters*8, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
    x12 = Conv2D(nfilters*8, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x12)


    #rgb depth fusion
    x = Add()([x10, x12])
    x = Conv2D(nfilters*8, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
    x = Conv2D(nfilters*8, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    
    x13 = Conv2D(nfilters*64, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x11)
    #esto agregue
    x13 = Conv2D(nfilters*64, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x13)
    x13 = Dropout(0.5)(x13)
    x14 = Conv2D(nfilters*64, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
    x14 = Conv2D(nfilters*64, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x14)
    x14 = Dropout(0.5)(x14)
    
    x = Add()([x13, x14])
    x = Dropout(0.5)(x)  

    x = Conv2DTranspose(filters=nfilters*8, kernel_size=(4,4), strides=(2,2),padding='same')(x)
    x = concatenate([Add()([x10, x12]), x], axis=3)
    x = Conv2D(nfilters*8, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
  
    x = Conv2DTranspose(filters=nfilters*4, kernel_size=(4,4), strides=(2,2),padding='same')(x)
    x = concatenate([Add()([x7, x9]), x], axis=3)
    x = Conv2D(nfilters*4, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)

    x = Conv2DTranspose(filters=nfilters*2, kernel_size=(4,4), strides=(2,2),padding='same')(x)
    x = concatenate([Add()([x4, x6]), x], axis=3)
    x = Conv2D(nfilters*2, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
   
    x = Conv2DTranspose(filters=nfilters*1, kernel_size=(4,4), strides=(2,2),padding='same')(x)
    x = Conv2D(nfilters*1, kernel_size =(size_kernel, size_kernel), strides =(size_stride, size_stride),dilation_rate=dilation, activation="relu", padding="same")(x)
    x = Conv2D(filters=nclasses, kernel_size=(size_kernel, size_kernel), padding='same')(x)
    x = Reshape((img_height*img_width, nclasses))(x)
    x = Activation('softmax')(x)
    model = Model(inputs=[input1, input2], outputs=x, name='RedNet')
    print('. . . . .Building network successful. . . . .')
    return model
