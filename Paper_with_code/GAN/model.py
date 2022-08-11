import tensorflow as tf
from tensorflow.keras import models, layers, utils
from tensorflow.keras import backend as K

class BuildModel():
    def __init__(self, img_shape, z_dim, label_dim=None):
        self.img_shape = img_shape
        self.z_dim = z_dim
        self.label_dim = label_dim

    def build_gene(self, activation = 'selu',last_activation='sigmoid', kernel_size=5):
        h, w, ch = self.img_shape
        z = layers.Input(shape=[self.z_dim,], name="noise")
        if self.label_dim:
            c = layers.Input(shape=[self.label_dim,], name='condition')
            y = layers.concatenate([z, c])
            y = layers.Dense(int(w/4)*int(h/4)*128)(y)
        else:
            y = layers.Dense(int(w/4)*int(h/4)*128)(z)
        y = layers.Reshape( [int(w/4),int(h/4),128] )(y)
        y = layers.BatchNormalization()(y)
        y = layers.Conv2DTranspose(64, kernel_size=5, padding='same', strides=2, activation=activation)(y)
        y = layers.BatchNormalization()(y)
        y = layers.Conv2DTranspose(ch, kernel_size=5, padding='same', strides=2, activation=last_activation)(y)
        
        if self.label_dim:
            model = models.Model([z, c], y, name='Generator')
        else:
            model = models.Model(z, y, name='Generator')
        return model

    def build_disc(self,activation='relu',last_activation='sigmoid', kernel_size=5):
        h, w, ch = self.img_shape
        x = layers.Input(shape=self.img_shape, name='image')
        
        def _condition_vector(x):
            y = K.expand_dims(x, axis=1)
            y = K.expand_dims(y, axis=1)
            y = K.tile(y, [1, h, w, 1])
            return y
        
        if self.label_dim:
            c = layers.Input(shape= self.label_dim, name='condition')
            c = layers.Lambda(_condition_vector)(c)
            y = layers.concatenate([x, c], axis=3)
            y = layers.Conv2D(64, kernel_size=kernel_size, strides=2, padding='same', activation=activation)(y)
        else:
            y = layers.Conv2D(64, kernel_size=kernel_size, strides=2, padding='same', activation=activation)(x)
        y = layers.Dropout(.5)(y)
        y = layers.Conv2D(128, kernel_size=kernel_size, strides=2, padding='same', activation=activation)(y)
        y = layers.Dropout(.5)(y)
        y = layers.Flatten()(y)

        y = layers.Dense(1, activation=last_activation)(y)
        if self.label_dim:
            model = models.Model([x,c], y, name='Discriminator')
        else:
            model = models.Model(x, y, name='Discriminator')
        return model