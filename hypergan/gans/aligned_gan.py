import importlib
import json
import numpy as np
import os
import sys
import time
import uuid
import copy

from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.samplers import *
from hypergan.trainers import *

import hyperchamber as hc
from hyperchamber import Config
from hypergan.ops import TensorflowOps
import tensorflow as tf
import hypergan as hg

from hypergan.gan_component import ValidationException, GANComponent
from .base_gan import BaseGAN

from hypergan.distributions.uniform_distribution import UniformDistribution
from hypergan.trainers.experimental.consensus_trainer import ConsensusTrainer

class AlignedGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)

    def required(self):
        """
        `input_encoder` is a discriminator.  It encodes X into Z
        `discriminator` is a standard discriminator.  It measures X, reconstruction of X, and G.
        `generator` produces two samples, input_encoder output and a known random distribution.
        """
        return "generator discriminator ".split()

    def create(self):
        config = self.config
        ops = self.ops

        with tf.device(self.device):
            def random_like(x):
                return UniformDistribution(self, config.latent, output_shape=self.ops.shape(x)).sample
            self.latent = self.create_component(config.latent, name='latent')


            #zgb = self.create_component(config.encoder, input=self.inputs.xb, name='encoder_b')
            zga = self.create_component(config.encoder, input=self.inputs.xa, name='encoder_a')
            zgb = self.create_component(config.encoder, input=self.inputs.xb, name='encoder_b')

            self.zgb = zgb
            self.zga = zga

            rzb = random_like(zgb.sample)

            g_ab = self.create_component(config.generator, input=zga.sample, name='b_generator', reuse=False, context={"input": self.inputs.xa})
            g_ba = self.create_component(config.generator, input=zgb.sample, name='a_generator', reuse=False, context={"input": self.inputs.xb})

            random_gb = self.create_component(config.generator, input=rzb, name='b_generator', reuse=True)
            random_ga = self.create_component(config.generator, input=random_like(self.zga.sample), name='a_generator', reuse=True)

            ga = g_ba
            gb = g_ab

            xa_hat = g_ab.sample
            xb_hat = g_ba.sample

            self.ga = g_ba
            self.gb = g_ab

            self.uniform_sample = gb.sample

            xba = ga.sample
            xab = gb.sample
            xa = self.inputs.xa
            xb = self.inputs.xb


            t0 = xb
            t1 = g_ab.sample
            f0 = zgb.sample
            f1 = zga.sample

            self.f1 = f1
            self.f0 = f0
            stack = [t0, t1]
            stacked = ops.concat(stack, axis=0)
            features = ops.concat([f0, f1], axis=0)
            self.features = features
            ugb = gb.sample
            zub = zgb.sample
            sourcezub = zgb.sample

            skip_connections = []
            d = self.create_component(config.discriminator, name='d_ab', 
                    skip_connections=skip_connections,
                    input=stacked, features=[features])

            self.skip_connections = skip_connections
            self.discriminator = d
            l = self.create_loss(config.loss, d, self.inputs.xa, ga.sample, len(stack))
            self.loss = l
            self.losses = [self.loss]
            self.standard_loss = l
            self.z_loss = l
            loss1 = l
            d_loss1 = l.d_loss
            g_loss1 = l.g_loss

            d_vars1 = d.variables()
            g_vars1 = gb.variables()+zga.variables()+zgb.variables()

            self.generator = g_ab

            d_loss = l.d_loss
            g_loss = l.g_loss


            metrics = {
                    'g_loss': l.g_loss,
                    'd_loss': l.d_loss
                }

            self._g_vars = g_vars1
            self._d_vars = d_vars1

            self.loss = hc.Config({
                'd_fake':l.d_fake,
                'd_real':l.d_real,
                'sample': [d_loss1, g_loss1],
                'metrics': metrics
                })
            if self.method == "sample":
                trainer = None
            else:
                trainer = self.create_component(config.trainer)

            self.initialize_variables()

        self.trainer = trainer
        self.encoder = hc.Config({"sample":ugb})
        self.uniform_distribution = hc.Config({"sample":zub})
        self.uniform_distribution_source = hc.Config({"sample":sourcezub})
        self.zb = zgb.sample
        self.z_hat = gb.sample
        self.x_input = self.inputs.xa
        self.autoencoded_x = xb_hat

        self.cyca = xa_hat
        self.cycb = xb_hat
        self.xba = xba
        self.xab = xab
        self.uga = ugb
        self.ugb = ugb

    def d_vars(self):
        return self._d_vars

    def g_vars(self):
        return self._g_vars

    def fitness_inputs(self):
        return [
            self.uniform_distribution.sample, self.inputs.xa
        ]

    def create_loss(self, loss_config, discriminator, x, generator, split):
        loss = self.create_component(loss_config, discriminator = discriminator, x=x, generator=generator, split=split)
        return loss

    def create_cycloss(self, x_input, x_hat):
        config = self.config
        ops = self.ops
        distance = config.distance or ops.lookup('l1_distance')
        cycloss_lambda = config.cycloss_lambda
        if cycloss_lambda is None:
            cycloss_lambda = 10

        cycloss = tf.reduce_mean(distance(x_input, x_hat))

        cycloss *= cycloss_lambda
        return cycloss

    def input_nodes(self):
        "used in hypergan build"
        if hasattr(self.generator, 'mask_generator'):
            extras = [self.mask_generator.sample]
        else:
            extras = []
        return extras + [
                self.x_input
        ]


    def output_nodes(self):
        "used in hypergan build"

        if hasattr(self.generator, 'mask_generator'):
            extras = [
                self.mask_generator.sample, 
                self.generator.g1x,
                self.generator.g2x
            ]
        else:
            extras = []
        return extras + [
                self.encoder.sample,
                self.generator.sample, 
                self.uniform_sample,
                self.generator_int
        ]
