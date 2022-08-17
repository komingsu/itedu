import tensorflow as tf
from tensorflow.keras import models, layers, utils
from utils import display_sample_img
import numpy as np

class GAN():

    def __init__(self, gene, disc, img_shape, noise_dims, label_dims=None):
        self.name = 'GAN'
        self.gene = gene
        self.disc = disc
        self.img_shape = img_shape
        self.noise_dims = noise_dims
        self.label_dims=label_dims
        

    def Compile(self, loss='binary_crossentropy', optimizer='adam'):

        self.disc.compile(loss=loss, optimizer=optimizer)
        self.disc.trainable = False

        input_noise = layers.Input(shape=self.noise_dims)
        if self.label_dims:
            input_label = layers.Input(shape=self.label_dims)
            fake_img = self.gene([input_noise, input_label])
            logit = self.disc([fake_img, input_label])
            self.gan = models.Model([input_noise, input_label], logit, name=self.name)
            self.gan.compile(loss=loss, optimizer=optimizer)
        else:
            fake_img = self.gene(input_noise)
            logit = self.disc(fake_img)
            self.gan = models.Model(input_noise, logit, name=self.name)
            self.gan.compile(loss=loss, optimizer=optimizer)

    def _make_datasets(self, x_data, y_data=None):
        if self.label_dims:
            dataset = tf.data.Dataset.from_tensor_slices((x_data,y_data)).shuffle(1)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(x_data).shuffle(1)
        dataset = dataset.batch(self.batch_size,drop_remainder=True).prefetch(1)
        return dataset

    def _make_constants(self):
        zeros = tf.constant([0.], shape=[self.batch_size, 1])
        ones = tf.constant([1.], shape=[self.batch_size, 1])
        return zeros, ones

    def _make_random(self):
        return tf.random.normal(shape=[self.batch_size, self.noise_dims])
    def _make_random_label(self):
        rnd_labels = np.random.randint(0, self.label_dims, self.batch_size)
        rnd_labels = utils.to_categorical(rnd_labels, self.label_dims)
        return rnd_labels

    def fit(self, x_data, y_data=None, epochs=1, batch_size=32, standard=False):

        # setting
        self.batch_size = batch_size
        if self.label_dims:
            train_ds = self._make_datasets(x_data, y_data)
            seed_noises = tf.random.normal(shape=[30, self.noise_dims])
            seed_labels =  np.tile(utils.to_categorical(np.arange(10), 10), (3,1))
        else:
            train_ds = self._make_datasets(x_data)
        zeros, ones = self._make_constants()

        # train
        history = {'d_loss': [], 'g_loss': []}
        for epoch in range(1 + epochs):
            if epoch > 0:
                for h in history:
                    history[h].append(0)
                
                if self.label_dims:
                    for real_imgs , real_labels in train_ds:
                        # phase 1 - training the discriminator
                        fake_imgs = self.gene.predict_on_batch( [self._make_random(), real_labels ])

                        self.disc.trainable = True
                        d_loss_real = self.disc.train_on_batch([real_imgs, real_labels], ones)
                        d_loss_fake = self.disc.train_on_batch([fake_imgs, real_labels], zeros)
                        d_loss = (0.5 * d_loss_real) + (0.5 * d_loss_fake)

                        # phase 2 - training the generator
                        self.disc.trainable = False
                        g_loss = self.gan.train_on_batch([self._make_random(), self._make_random_label()], ones)

                        history['d_loss'][-1] += d_loss
                        history['g_loss'][-1] += g_loss

                    # end 1 epoch
                    print('* epoch: %i, d_loss: %f, g_loss: %f' %
                          (epoch, history['d_loss'][-1], history['g_loss'][-1]))
                    fake_imgs = self.gene.predict([seed_noises, seed_labels])
                    display_sample_img(fake_imgs, (3, 10), standard=standard, size=2)
                else:
                    for real_imgs in train_ds:
                        # phase 1 - training the discriminator
                        fake_imgs = self.gene.predict_on_batch(self._make_random())

                        self.disc.trainable = True
                        d_loss_real = self.disc.train_on_batch(real_imgs, ones)
                        d_loss_fake = self.disc.train_on_batch(fake_imgs, zeros)
                        d_loss = (0.5 * d_loss_real) + (0.5 * d_loss_fake)

                        # phase 2 - training the generator
                        self.disc.trainable = False
                        g_loss = self.gan.train_on_batch(self._make_random(), ones)

                        history['d_loss'][-1] += d_loss
                        history['g_loss'][-1] += g_loss

                    # end 1 epoch
                    print('* epoch: %i, d_loss: %f, g_loss: %f' %
                          (epoch, history['d_loss'][-1], history['g_loss'][-1]))
                    fake_imgs = self.gene.predict(self._make_random())
                    display_sample_img(fake_imgs, (2, 8), standard=standard, size=2)