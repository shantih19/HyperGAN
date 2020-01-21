"""
The command line interface.  Trains a directory of data.
"""
import gc
import sys
import os
import functools
import hyperchamber as hc
import tensorflow as tf
import numpy as np
from hypergan.gan_component import ValidationException
from .inputs import *
from .viewer import GlobalViewer
from .configuration import Configuration
from tensorflow.contrib import tpu
import hypergan as hg
import time

import os
import shutil
import sys

from hypergan.losses.supervised_loss import SupervisedLoss
from hypergan.multi_component import MultiComponent
from time import sleep


class CLI:
    def __init__(self, args={}, gan_fn=None, inputs_fn=None, gan_config=None):
        self.samples = 0
        self.steps = 0
        self.gan_fn = gan_fn
        self.gan_config = gan_config
        self.inputs_fn = inputs_fn
        self.gan = gan_fn(gan_config)
        args = hc.Config(args)
        self.args = args

        crop = self.args.crop

        self.config_name = self.args.config or 'default'
        self.method = args.method or 'test'
        self.total_steps = args.steps or -1
        self.sample_every = self.args.sample_every or 100

        self.sampler_name = args.sampler
        self.sampler = None

        self.validate()

        self.loss_every = self.args.loss_every or 1

        if self.args.save_losses:
            import matplotlib.pyplot as plt
            self.arr = []
            self.fig, self.ax = plt.subplots()
            self.temp = 0

        self.advSavePath = os.path.abspath("saves/" + self.config_name) + "/"
        if self.args.save_file:
            self.save_file = self.args.save_file
        else:
            default_save_path = os.path.abspath("saves/" + self.config_name)
            self.save_file = default_save_path + "/model.ckpt"
            self.create_path(self.save_file)

        title = "[hypergan] " + self.config_name
        GlobalViewer.set_options(
            enable_menu=self.args.menu,
            title=title,
            viewer_size=self.args.viewer_size,
            enabled=self.args.viewer,
            zoom=self.args.zoom)

    def sample(self, allow_save=True):
        """ Samples to a file.  Useful for visualizing the learning process.

        If allow_save is False then saves will not be created.

        Use with:

             ffmpeg -i samples/grid-%06d.png -vcodec libx264 -crf 22 -threads 0 grid1-7.mp4

        to create a video of the learning process.
        """
        sample_file = "samples/%s/%06d.png" % (self.config_name, self.samples)
        self.create_path(sample_file)
        self.lazy_create()
        sample_list = self.sampler.sample(sample_file, allow_save and self.args.save_samples)
        self.samples += 1

        return sample_list

    def validate(self):
        return True

    def lazy_create(self):
        if (self.sampler == None):
            self.sampler = self.gan.sampler_for(self.sampler_name)(self.gan, samples_per_row=self.args.width)
            if (self.sampler == None):
                raise ValidationException("No sampler found by the name '" + self.sampler_name + "'")

    def step(self):
        bgan = self.gan
        self.gan.step()
        if hasattr(self.gan, 'newgan') and bgan.destroy:
            self.sampler = None
            self.gan = self.gan.newgan
            gc.collect()
            refs = gc.get_referrers(bgan)
            d = bgan.trainer._delegate
            bgan.trainer = None
            gc.collect()
            del bgan
            tf.reset_default_graph()

            gc.collect()

        if self.steps % self.sample_every == 0 and self.args.sampler:
            sample_list = self.sample()

        self.steps += 1

        x = []
        if hasattr(self.gan.loss, "sample"):
            loss = self.gan.loss.sample
            if self.args.save_losses:
                temp2 = False
                if (len(self.arr) == 0):
                    for i in range(0, len(loss)):
                        self.arr.append([]);
                for i in range(0, len(loss)):
                    self.arr[i].append(self.gan.session.run(loss[i]))
                for j in range(0, len(self.arr)):
                    if (len(self.arr[j]) > 100):
                        self.arr[j].pop(0)
                        if (not temp2 == True):
                            self.temp += 1
                            temp2 = True
                if (temp2 == True):
                    temp2 = False
        else:
            if (self.args.save_losses):
                temp2 = False
                if (len(self.arr) == 0):
                    for i in range(0, len(self.gan.trainer.losses)):
                        self.arr.append([]);
                for i in range(0, len(self.gan.trainer.losses)):
                    self.arr[i].append(self.gan.session.run(self.gan.trainer.losses[i][1]))
                for j in range(0, len(self.arr)):
                    if (len(self.arr[j]) > 100):
                        self.arr[j].pop(0)
                        if (not temp2 == True):
                            self.temp += 1
                            temp2 = True
                if (temp2 == True):
                    temp2 = False
        if (self.args.save_losses and self.steps % self.loss_every == 0):
            for i in range(0, len(self.arr)):
                x2 = []
                for j in range(self.temp, self.temp + len(self.arr[i])):
                    x2.append(j)
                x.append(x2)
            self.ax.cla()
            for i in range(0, len(self.arr)):
                self.ax.plot(x[i], self.arr[i])
            self.ax.grid()
            self.ax.title.set_text("HyperGAN losses")
            self.ax.set_xlabel('Steps')
            self.ax.set_ylabel('Losses')
            self.create_path("losses/" + self.config_name + "/%06d.png" % (self.steps))
            self.fig.savefig("losses/" + self.config_name + "/%06d.png" % (self.steps))

    def create_path(self, filename):
        return os.makedirs(os.path.expanduser(os.path.dirname(filename)), exist_ok=True)

    def build(self):
        return self.gan.build()

    def serve(self, gan):
        return gan_server(self.gan.session, config)

    def sample_forever(self):
        while not self.gan.destroy:
            self.sample()

    def train_tpu(self):
        i = 0
        tpu_name = self.args.device.split(":")[1]
        cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=tpu_name)

        config = tf.compat.v1.estimator.tpu.RunConfig(
            cluster=cluster_resolver,
            model_dir=self.save_file,
            tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
                num_shards=8,
                iterations_per_loop=10))

        def input_fn(mode, params):
            batch_size = params["batch_size"]
            inputs = self.inputs_fn(batch_size=batch_size)
            x = inputs.dataset.make_one_shot_iterator().get_next()
            x = tf.reshape(x, [batch_size, 64, 64, 3])
            return x, x

        spec = {'image/encoded': tf.FixedLenFeature([], tf.string)}

        def serving_input_fn():
            serialized_tf_example = tf.placeholder(dtype=tf.string, shape=None,
                                                   name='image/encoded')
            # key (e.g. 'examples') should be same with the inputKey when you 
            # buid the request for prediction
            receiver_tensors = {'examples': serialized_tf_example}
            features = tf.parse_example(serialized_tf_example, spec)
            return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

        self.gan = self.gan_fn(self.gan_config, None)
        est = self.gan.tpu_gan_estimator(config)

        config = tf.ConfigProto()
        cluster_spec = cluster_resolver.cluster_spec()
        if cluster_spec:
            config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())

        with tf.Session(cluster_resolver.master(), config=config) as session:

            self.gan.session = session
            for v in self.gan.variables():
                session.run(v.initializer)
            if not self.gan.load(self.save_file):
                print("Initializing new model")
            else:
                print("Model loaded")
            while ((i < self.total_steps or self.total_steps == -1)):
                if i % 1 == 0:
                    print("Step ", i)
                # if i % 100 == 0:
                #    self.sample()
                i += 1
                est.train(input_fn, max_steps=10)

                if (self.args.save_every != None and
                        self.args.save_every != -1 and
                        self.args.save_every > 0 and
                        i % self.args.save_every == 0):
                    print(" |= Saving network")
                    est.export_saved_model(self.save_file, serving_input_receiver_fn=serving_input_fn)
                if self.args.ipython:
                    self.check_stdin()
            session.run(tpu.shutdown_system())

    def train(self):
        i = 0
        if (self.args.ipython):
            import fcntl
            fd = sys.stdin.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        if "tpu" in self.args.device:
            self.train_tpu()
            return

        while ((i < self.total_steps or self.total_steps == -1) and not self.gan_fn().destroy):
            i += 1
            start_time = time.time()
            self.step()

            if (self.args.save_every != None and
                    self.args.save_every != -1 and
                    self.args.save_every > 0 and
                    i % self.args.save_every == 0):
                print(" |= Saving network")
                self.gan.save(self.save_file)
                self.create_path(self.advSavePath + 'advSave.txt')
                with open(self.advSavePath + 'advSave.txt', 'w') as the_file:
                    the_file.write(str(self.steps) + "\n")
                    the_file.write(str(self.samples) + "\n")
            if self.args.ipython:
                self.check_stdin()
            end_time = time.time()

    def check_stdin(self):
        try:
            input = sys.stdin.read()
            if input[0] == "y":
                return
            from IPython import embed
            # Misc code
            embed()

        except:
            return

    def new(self):
        template = self.args.directory + '.json'
        print(
            "[hypergan] Creating new configuration file '" + template + "' based off of '" + self.config_name + ".json'")
        if os.path.isfile(template):
            raise ValidationException("File exists: " + template)
        source_configuration = Configuration.find(self.config_name + ".json")
        shutil.copyfile(source_configuration, template)

        return

    def add_supervised_loss(self):
        if self.args.classloss:
            print("[discriminator] Class loss is on.  Semi-supervised learning mode activated.")
            supervised_loss = SupervisedLoss(self.gan, self.gan.config.loss)
            self.gan.loss = MultiComponent(components=[supervised_loss, self.gan.loss], combine='add')
            # EWW
        else:
            print("[discriminator] Class loss is off.  Unsupervised learning mode activated.")

    def run(self):
        if self.method == 'train':
            self.train()
        elif self.method == 'build':
            if not self.gan.load(self.save_file):
                raise ValidationException("Could not load model: " + self.save_file)
            else:
                with open(self.advSavePath + 'advSave.txt', 'r') as the_file:
                    content = [x.strip() for x in the_file]
                    self.steps = int(content[0])
                    self.samples = int(content[1])
                print("Model loaded")
            self.build()
        elif self.method == 'new':
            self.new()
        elif self.method == 'sample':
            self.add_supervised_loss()
            if not self.gan.load(self.save_file):
                print("Initializing new model")
            else:
                with open(self.advSavePath + 'advSave.txt', 'r') as the_file:
                    content = [x.strip() for x in the_file]
                    self.steps = int(content[0])
                    self.samples = int(content[1])
                print("Model loaded")

            tf.train.start_queue_runners(sess=self.gan.session)
            self.sample_forever()
            tf.reset_default_graph()
            self.gan.session.close()
        elif self.method == 'test':
            print("Hooray!")
            print("Hypergan is installed correctly.  Testing tensorflow for GPU support.")
            with tf.Session() as sess:
                devices = sess.list_devices()

            if not tf.test.gpu_device_name():
                print("Warning: no default GPU device available")
                allgood = False
            else:
                print("Default GPU is available")
                allgood = True
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
            print("Current available tensorflow devices:")
            for device in devices:
                print(device)
            if allgood:
                print(
                    "Congratulations!  Tensorflow and hypergan both look installed correctly.  If you still experience issues come let us know on discord.")
            else:
                print("There were errors in the test, please see the logs")
