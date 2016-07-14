import gym
import numpy as np
import tensorflow as tf
import time
import os

from itertools import count

from tf_rl.models import Layer, LambdaLayer, ConvLayer, SeqLayer
from tf_rl.controller.discrete_deepq import DiscreteDeepQ

# CRAZY VARIABLES
REAL_TIME            = False
RENDER               = True

ENVIRONMENT          = "Pong-v0"
MODEL_SAVE_DIR       = "./{}-model/".format(ENVIRONMENT)
MODEL_SAVE_EVERY_S   = 60

# SENSIBLE VARIABLES
FPS         = 60
MAX_FRAMES  = 1000
IMAGE_SHAPE = (210, 160, 3)
OBS_SHAPE   = (210, 160, 6)
NUM_ACTIONS = 6


def make_model():
    """Create a tensorflow convnet that takes image as input
    and outputs a predicted discounted score for every action"""

    with tf.variable_scope('convnet'):
        convnet = SeqLayer([
            ConvLayer(3, 3,   6, 32,  stride=(1,1), scope='conv1'),    # out.shape = (B, 210, 160, 32)
            LambdaLayer(tf.nn.sigmoid),
            ConvLayer(2, 2,  32, 64,  stride=(2,2), scope='conv2'),    # out.shape = (B, 105, 80, 64)
            LambdaLayer(tf.nn.sigmoid),
            ConvLayer(3, 3,  64, 64,  stride=(1,1), scope='conv3'),    # out.shape = (B, 105, 80, 64)
            LambdaLayer(tf.nn.sigmoid),
            ConvLayer(2, 2,  64, 128, stride=(2,2), scope='conv4'),    # out.shape = (B, 53, 40, 128)
            LambdaLayer(tf.nn.sigmoid),
            ConvLayer(3, 3, 128, 128, stride=(1,1), scope='conv5'),    # out.shape = (B, 53, 40, 128)
            LambdaLayer(tf.nn.sigmoid),
            ConvLayer(2, 2, 128, 256, stride=(2,2), scope='conv6'),    # out.shape = (B, 27, 20, 256)
            LambdaLayer(tf.nn.sigmoid),
            LambdaLayer(lambda x: tf.reshape(x, [-1, 27 * 20 * 256])), # out.shape = (B, 27 * 20 * 256)
            Layer(27 * 20 * 256, 6, scope='proj_actions')              # out.shape = (B, 6)
        ], scope='convnet')
    return convnet


def make_controller():
    """Create a deepq controller"""
    session    = tf.Session()

    model = make_model()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    return DiscreteDeepQ(OBS_SHAPE,
                         NUM_ACTIONS,
                         model,
                         optimizer,
                         session,
                         random_action_probability=0.1,
                         minibatch_size=8,
                         discount_rate=0.99,
                         exploration_period=500000,
                         max_experience=10000,
                         target_network_update_rate=0.01,
                         store_every_nth=4,
                         train_every_nth=4)

# TODO(szymon): apparently both DeepMind and Karpathy
# people normalize their frames to sizes 80x80 and grayscale.
# should we do this?
def normalize_frame(o):
    """Change from uint in range (0, 255) to float in range (0, 1)"""
    return o.astype(np.float32) / 255.0

def main():
    env        = gym.make(ENVIRONMENT)
    controller = make_controller()

    # Load existing model.
    if os.path.exists(MODEL_SAVE_DIR):
        print "loading model... ",
        controller.restore(MODEL_SAVE_DIR)
        print 'done.'
    last_model_save = time.time()

    # For every game
    for game_no in count():
        # Reset simulator
        frame_tm1         = normalize_frame(env.reset())
        frame_t, _, _, _  = env.step(env.action_space.sample())
        frame_t           = normalize_frame(frame_t)

        rewards = []
        for _ in range(MAX_FRAMES):
            start_time = time.time()

            # observation consists of two last frames
            # this is important so that we can detect speed.
            observation_t = np.concatenate([frame_tm1, frame_t], 2)

            # pick an action according to Q-function learned so far.
            action      = controller.action(observation_t)

            if RENDER: env.render()

            # advance simulator
            frame_tp1, reward, done, info =  env.step(action)
            frame_tp1 = normalize_frame(frame_tp1)
            if done: break

            observation_tp1 = np.concatenate([frame_t, frame_tp1], 2)

            # store transitions
            controller.store(observation_t, action, reward, observation_tp1)
            # run a single iteration of SGD
            controller.training_step()


            frame_tm1, frame_t = frame_t, frame_tp1
            rewards.append(reward)

            # if real time visualization is requested throttle down FPS.
            if REAL_TIME:
                time_passed = time.time() - start_time
                time_left   = 1.0 / FPS - time_passed

                if time_left > 0:
                    time.sleep(time_left)

            # save model if time since last save is greater than
            # MODEL_SAVE_EVERY_S
            if time.time() - last_model_save >= MODEL_SAVE_EVERY_S:
                if not os.path.exists(MODEL_SAVE_DIR):
                    os.makedirs(MODEL_SAVE_DIR)
                controller.save(MODEL_SAVE_DIR, debug=True)
                last_model_save = time.time()

        # Count scores. This relies on specific score values being
        # assigned by openai gym and might break in the future.
        points_lost = rewards.count(-1.0)
        points_won  = rewards.count(1.0)
        exploration_done = controller.exploration_completed()

        print "Game no %d is over. Exploration %.1f done. Points lost: %d, points won: %d" % \
                (game_no, 100.0 * exploration_done, points_lost, points_won)

if __name__ == '__main__':
    main()

