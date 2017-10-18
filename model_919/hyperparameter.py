import tensorflow as tf
from other.nn_func import leaky_relu
import tensorflow.contrib.layers as ly

# Created in nn_func.py, instead of importing from here.
# WEIGHT_INITIALIZER = tf.truncated_normal_initializer(stddev=0.01)

# Optimizer hyper parameters for discriminator
OPTIMIZER_DIS_HP = {'learning_rate': 0.0001,
                    'optimizer': tf.train.AdamOptimizer,
                    'lambda': 10,  # coefficient when compute gradient penalty, only for Dis optmzr
                    'clip': 0.01
                    }

OPTIMIZER_GEN_HP = {'learning_rate': 0.0001,
                    'optimizer': tf.train.AdamOptimizer,
                    }

# Generator hyper parameters
GENERATOR_HP = {'z_dim': 128,
                'gf_dim': 64,
                # 'emb_reduced_dim': 256,
                # 'kernel_size': 3,
                # 'stride': 2,
                'act_fn': tf.nn.relu,
                'norm_fn': ly.batch_norm,
                'norm_params': {'is_training': True, 'updates_collections': None},

                # 'normalizer_fn': None,
                # 'weights_initializer': tf.truncated_normal_initializer(stddev=0.01)
                }

# Discriminator hyper parameters
DISCRIMINATOR_HP = {'df_dim': 64,
                    'emb_reduced_dim': 8 * 64,
                    # 'kernel_size': 3,
                    # 'stride': 2,
                    'act_fn': leaky_relu,
                    'norm_fn': None,
                    'norm_params': None
                    # 'weights_initializer': tf.truncated_normal_initializer(stddev=0.01),
                    }
