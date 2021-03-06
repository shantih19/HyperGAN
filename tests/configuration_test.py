import tensorflow as tf
import hyperchamber as hc
import numpy as np
import hypergan as hg
from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

class ConfigurationTest(tf.test.TestCase):
    def test_constructor(self):
        with self.test_session():
            default = hg.Configuration.default()
            self.assertNotEqual(default.trainer, None)
            self.assertNotEqual(default.discriminator, None)
            self.assertNotEqual(default.loss, None)

if __name__ == "__main__":
    tf.test.main()
