import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1' # '0' for DEBUG=all [default], '1' to filter INFO msgs, '2' to filter WARNING msgs, '3' to filter all msgs
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import tensorflow
#physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
#tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)


import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from scipy.interpolate import make_interp_spline

img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)

z_dim = 100 #size of noise vector / latent space 

def build_generator(img_shape, z_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(img_rows*img_cols*channels, activation='tanh'))
    model.add(Reshape(img_shape))
    return model

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

generator = build_generator(img_shape, z_dim)

discriminator.trainable = False  #apply only on gan !
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())



losses = []
accuracies = []
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval,path_1):
    (X_train, _), (_, _) = mnist.load_data()
    X_train = X_train / 127.5 - 1.0    #rescale to [-1; 1] (because of 'tanh')
    X_train = np.expand_dims(X_train, axis=3) 
    
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for iteration in range(iterations):
        idx = np.random.randint(0, X_train.shape[0], batch_size)  #B:TODO: no garantee that we see all the images !!!
        imgs = X_train[idx]
    
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)
    
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        z = np.random.normal(0, 1, (batch_size, z_dim))    
        gen_imgs = generator.predict(z)
        
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)
            print("%d {D loss: %f, acc: %.2f%%] [G loss:%f]" % (iteration+1, d_loss, 100.0 * accuracy, g_loss))
            sample_images(generator, path_1)

image_grid_rows=1
image_grid_columns=13
# =============================================================================
# fig = plt.figure(figsize=(image_grid_rows, image_grid_columns))
# fig.show()
# #reuse the same noise vector to visualise progression over time
# =============================================================================
z_sample_images = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

fig = plt.figure(figsize=(image_grid_rows,image_grid_columns)) # Notice the equal aspect ratio
ax = [fig.add_subplot(5,13,i+1) for i in range(13)]

for a in ax:
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.set_aspect('equal')

fig.subplots_adjust(wspace=0, hspace=0)
plt.show()

def sample_images(generator, path_1):
    #z_sample_images = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
    gen_imgs = generator.predict(z_sample_images)
    for i in range(gen_imgs.shape[0]):
        plt.subplot(image_grid_rows, image_grid_columns, i+1)
        plt.imshow(gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    fig.canvas.draw()
    plt.pause(0.01)
    #plt.savefig(path_1+'/'+str(i)+'.png' )
    

# =============================================================================
# Creating paths for new directories
path = os.getcwd()
path_1 = path+"\\Exercise1"
#
# Checking existence or creating directories for report images
# if os.path.exists(path_1):
#    print(path_1 + ' : exists')
#else:
#    print(path_1 + ' : created')
#    os.mkdir(path_1)
# =============================================================================



iterations = 20000
batch_size = 128
sample_interval = 250

train(iterations, batch_size, sample_interval, path_1)



losses = np.array(losses)

# Plot training losses for Discriminator and Generator
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses.T[0], label="Discriminator loss")
plt.plot(iteration_checkpoints, losses.T[1], label="Generator loss")
x_new = np.linspace(0, iterations, 9)


plt.xticks(x_new, rotation=45)
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()


accuracies = np.array(accuracies)

# Plot Discriminator accuracy
plt.figure(figsize=(10, 5))
plt.plot(iteration_checkpoints, accuracies, label="Discriminator accuracy")


x_new = np.linspace(0, iterations, 9)

a_BSpline = make_interp_spline(iteration_checkpoints, accuracies)
y_new = a_BSpline(x_new)

plt.plot(x_new, y_new, '--')
plt.xticks(x_new, rotation=45)
plt.yticks(range(0, 100, 10))

plt.title("Discriminator Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.legend(['Discriminator accuracy', 'Tendency'])

plt.show()
