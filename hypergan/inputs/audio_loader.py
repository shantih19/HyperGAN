import glob
from hypergan.inputs.resize_audio_patch import resize_audio_with_crop_or_pad
from tensorflow.contrib import ffmpeg
import tensorflow as tf

class AudioLoader:
    """
    AudioLoader loads a set of mp3 files into a tensorflow input pipeline.
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
      directory = directories[0]
      seconds=height
      bitrate=width

      filenames = glob.glob(directory+"/**/*."+format)
      num_examples_per_epoch = 10000


      filenames = tf.convert_to_tensor(filenames, dtype=tf.string)

      input_queue = tf.train.slice_input_producer([filenames])

      # Read examples from files in the filename queue.
      value = tf.read_file(input_queue[0])
      #preprocess = tf.read_file(input_queue[0]+'.preprocess')

      #print("Loaded data", data)

      min_fraction_of_examples_in_queue = 0.4
      min_queue_examples = int(num_examples_per_epoch *
                               min_fraction_of_examples_in_queue)

      #data = tf.cast(data, tf.float32)
      data = ffmpeg.decode_audio(value, file_format=format, samples_per_second=bitrate, channel_count=channels)
      data = resize_audio_with_crop_or_pad(data, seconds*bitrate*channels, 0,True)
      #data = tf.slice(data, [0,0], [seconds*bitrate, channels])
      tf.Tensor.set_shape(data, [seconds*bitrate, channels])
      #data = tf.minimum(data, 1)
      #data = tf.maximum(data, -1)
      data = data/tf.reduce_max(tf.reshape(tf.abs(data),[-1]))
      print("DATA IS", data)
      self.x=self._get_data(data, min_queue_examples, self.batch_size)

      self.xa = self.x
      self.xb = self.x

      self.datasets = [self.x]


    def _get_data(self, image, min_queue_examples, batch_size):
      num_preprocess_threads = 1
      images= tf.train.shuffle_batch(
          [image],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity= 502,
          min_after_dequeue=128)
      return images

