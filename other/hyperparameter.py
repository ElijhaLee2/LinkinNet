import tensorflow as tf
import tensorflow.contrib.layers as ly
from other.function import leaky_relu

# Optimizer hyper parameters for discriminator
OPTIMIZER_DIS_HP = {'learning_rate': 0.0002,
                    'optimizer': tf.train.AdamOptimizer,
                    'lambda': 8,  # coefficient when compute gradient penalty, only for Dis optmzr
                    }

# Generator hyper parameters
GENERATOR_HP = {'z_dim': 128,
                'base_channel': 32,
                'kernel_size': 3,
                'stride': 2,
                'activation_fn': leaky_relu,
                'normalizer_fn': ly.batch_norm,
                'weights_initializer': tf.truncated_normal_initializer(stddev=0.01)}

# Discriminator hyper parameters
DISCRIMINATOR_HP = {'base_channel': 32,
                    'kernel_size': 3,
                    'stride': 2,
                    'activation_fn': leaky_relu,
                    'normalizer_fn': None,
                    'weights_initializer': tf.truncated_normal_initializer(stddev=0.01),
                    'biases_initializer': tf.zeros_initializer,
                    }
