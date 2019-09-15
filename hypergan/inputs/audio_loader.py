import glob
import os
from natsort import natsorted, ns
from hypergan.inputs.resize_audio_patch import resize_audio_with_crop_or_pad
from tensorflow.contrib import ffmpeg
from hypergan.gan_component import ValidationException, GANComponent
import tensorflow as tf

class AudioLoader:
    """
    AudioLoader loads a set of mp3/wav files into a tensorflow input pipeline.
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_labels(dirs):
      next_id=0
      labels = {}
      for dir in dirs:
        labels[dir.split('/')[-1]]=next_id
        next_id+=1
      return labels,next_id

    def create(self, directories, format='mp3', width=16384, height=30, channels=2, crop=None, sequential=None, resize=None):
        self.datasets = []
        directory = directories[0]
        seconds=height
        bitrate=width
        for directory in directories:
            dirs = glob.glob(directory+"/*")
            dirs = [d for d in dirs if os.path.isdir(d)]

            if(len(dirs) == 0):
                dirs = [directory] 

            # Create a queue that produces the filenames to read.
            if(len(dirs) == 1):
                # No subdirectories, use all the images in the passed in path
                filenames = glob.glob(directory+"/*."+format)
            else:
                filenames = glob.glob(directory+"/**/*."+format)

            filenames = natsorted(filenames)

            print("[loader] AudioLoader found", len(filenames))
            self.file_count = len(filenames)
            if self.file_count == 0:
                raise ValidationException("No audio files found in '" + directory + "'")
            filenames = tf.convert_to_tensor(filenames, dtype=tf.string)

            def parse_function(filename):
                image_string = tf.read_file(filename)
                #if format == 'mp3' or format == 'wav':
                #    image = ffmpeg.decode_audio(image_string, file_format=format, samples_per_second=bitrate, channel_count=channels)
                #    image = resize_audio_with_crop_or_pad(image, seconds*bitrate*channels, 0,True)
                #else:
                #    print("[loader] Failed to load format", format)

                image = tf.zeros([height*width, channels])
                image = tf.cast(image, tf.float32)
                #tf.Tensor.set_shape(image, [height*width,channels])

                return image

            # Generate a batch of images and labels by building up a queue of examples.
            dataset = tf.data.Dataset.from_tensor_slices(filenames)
            if not sequential:
                print("Shuffling data")
                dataset = dataset.shuffle(self.file_count)
            dataset = dataset.map(parse_function, num_parallel_calls=4)
            dataset = dataset.batch(self.batch_size, drop_remainder=True)

            dataset = dataset.repeat()
            dataset = dataset.prefetch(1)

            self.datasets.append(tf.reshape(dataset.make_one_shot_iterator().get_next(), [self.batch_size, height*width, channels]))

            self.xs = self.datasets
            self.xa = self.datasets[0]
            if len(self.datasets) > 1:
                self.xb = self.datasets[1]
            else:
                self.xb = self.datasets[0]
            self.x = self.datasets[0]
            return self.xs

