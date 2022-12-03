import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import os
from utility import *
from PIL import Image

latent_dim = 64 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='sigmoid'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def get_frames(video, video_num):
    cap = cv2.VideoCapture(video)
    i = 0
    # a variable to set how many frames you want to skip
    frame_skip = 10
    # a variable to keep track of the frame to be saved
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i > frame_skip - 1:
            frame_count += 1
            path = 'train_images'
            final_path = os.path.join(path , video_num + '_test_'+str(frame_count*frame_skip)+'.jpg')
            cv2.imwrite(final_path, frame)
            i = 0
            continue
        i += 1

    cap.release()
    cv2.destroyAllWindows()

def get_100_videos_to_batches():
  path = "data/videos/training_data"
  videos = (os.listdir("data/videos/training_data"))
  count = 0
  for video_path in videos:
    if (count == 100):
      break
    video_num = video_path[:-4] # for naming purposes
    get_frames(path + '/' + video_path, video_num)
    count += 1

def train_vae():
  # get_100_videos_to_batches() already generated

  batch_size = 1
  img_height = 180
  img_width = 180
  data_dir = "/Users/linseyszabo/Desktop/dl_project_code/10417-project"

  train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

  epochs = 20  # epochs: Number of iterations for which training will be performed
  autoencoder = Autoencoder(latent_dim)
  autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

  autoencoder.fit(train_ds,
                epochs=epochs,
                shuffle=True,
                validation_data=val_ds)

if __name__ == '__main__':
  train_vae()