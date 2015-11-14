import tensorflow as tf
import random

from collections import deque

class DiscreteDeepQ(object):
    def __init__(self, observation_size,
                       num_actions,
                       observation_to_actions,
                       optimizer,
                       session,
                       random_action_probability=0.05,
                       exploration_period=1000,
                       minibatch_size=32,
                       discount_rate=0.95,
                       max_experience=30000,
                       summary_writer=None):
        """Initialized the Deepq object.

        Based on:
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        Parameters
        -------
        observation_size : int
            length of the vector passed as observation
        num_actions : int
            number of actions that the model can execute
        observation_to_actions: dali model
            model that implements activate function
            that can take in observation vector or a batch
            and returns scores (of unbounded values) for each
            action for each observation.
            input shape:  [batch_size, observation_size]
            output shape: [batch_size, num_actions]
        optimizer: tf.solver.*
            optimizer for prediction error
        session: tf.Session
            session on which to execute the computation
        random_action_probability: float (0 to 1)
        exploration_period: int
            probability of choosing a random
            action (epsilon form paper) annealed linearly
            from 1 to random_action_probability over
            exploration_period
        minibatch_size: int
            number of state,action,reward,newstate
            tuples considered during experience reply
        dicount_rate: float (0 to 1)
            how much we care about future rewards.
        max_experience: int
            maximum size of the reply buffer
        summary_writer: tf.train.SummaryWriter
            writer to log metrics
        """
        # memorize arguments
        self.observation_size          = observation_size
        self.num_actions               = num_actions

        self.observation_to_actions    = observation_to_actions
        self.optimizer                 = optimizer
        self.s                         = session

        self.random_action_probability = random_action_probability
        self.exploration_period        = exploration_period
        self.minibatch_size            = minibatch_size
        self.discount_rate             = tf.constant(discount_rate)
        self.max_experience            = max_experience

        # deepq state
        self.actions_executed_so_far = 0
        self.experience = deque()

        self.iteration = 0
        self.summary_writer = summary_writer

        self.create_variables()

    def linear_annealing(self, n, total, p_initial, p_final):
        """Linear annealing between p_initial and p_final
        over total steps - computes value at step n"""
        if n >= total:
            return p_final
        else:
            return p_initial - (n * (p_initial - p_final)) / (total)

    def create_variables(self):
        # FOR REGULAR ACTION SCORE COMPUTATION
        with tf.name_scope("observation"):
            self.observation        = tf.placeholder(tf.float32, (None, self.observation_size), name="observation")
            self.action_scores      = self.observation_to_actions(self.observation)
            self.predicted_actions  = tf.argmax(self.action_scores, dimension=1, name="predicted_actions")

        with tf.name_scope("future_rewards"):
            # FOR PREDICTING TARGET FUTURE REWARDS
            self.observation_mask     = tf.placeholder(tf.float32, (None,), name="observation_mask")
            self.rewards              = tf.placeholder(tf.float32, (None,), name="rewards")
            target_values             = tf.reduce_max(self.action_scores, reduction_indices=[1,]) * self.observation_mask
            self.future_rewards       = self.rewards + self.discount_rate * target_values

        with tf.name_scope("q_value_precition"):
            # FOR PREDICTION ERROR
            self.action_mask                = tf.placeholder(tf.float32, (None, self.num_actions))
            self.masked_action_scores       = tf.reduce_sum(self.action_scores * self.action_mask, reduction_indices=[1,])
            self.precomputed_future_rewards = tf.placeholder(tf.float32, (None,))
            temp_diff                       = self.masked_action_scores - self.precomputed_future_rewards
            self.prediction_error           = tf.reduce_mean(tf.square(temp_diff))
            self.train_op                   = self.optimizer.minimize(self.prediction_error)

        self.metrics = [
            tf.scalar_summary("prediction_error", self.prediction_error)
        ]

    def action(self, observation):
        """Given observation returns the action that should be chosen using
        DeepQ learning strategy. Does not backprop."""
        assert len(observation.shape) == 1, \
                "Action is performed based on single observation."

        self.actions_executed_so_far += 1
        exploration_p = self.linear_annealing(self.actions_executed_so_far,
                                              self.exploration_period,
                                              1.0,
                                              self.random_action_probability)

        if random.random() < exploration_p:
            return random.randint(0, self.num_actions - 1)
        else:
            return self.s.run(self.predicted_actions, {self.observation: observation[np.newaxis,:]})[0]

    def store(self, observation, action, reward, newobservation):
        """Store experience, where starting with observation and
        execution action, we arrived at the newobservation and got the
        reward reward

        If newstate is None, the state/action pair is assumed to be terminal
        """
        self.experience.append((observation, action, reward, newobservation))
        if len(self.experience) > self.max_experience:
            self.experience.popleft()

    def training_step(self):
        """Pick a self.minibatch_size exeperiences from reply buffer
        and backpropage the value function.
        """
        if len(self.experience) <  self.minibatch_size:
            return

        # sample experience.
        samples   = random.sample(range(len(self.experience)), self.minibatch_size)
        samples   = [self.experience[i] for i in samples]

        # bach states
        states         = np.empty((len(samples), self.observation_size))
        newstates      = np.empty((len(samples), self.observation_size))
        action_mask    = np.zeros((len(samples), self.num_actions))

        newstates_mask = np.empty((len(samples),))
        rewards        = np.empty((len(samples),))

        for i, (state, action, reward, newstate) in enumerate(samples):
            states[i] = state
            action_mask[i] = 0
            action_mask[i][action] = 1
            rewards[i] = reward
            if newstate is not None:
                newstates[i] = state
                newstates_mask[i] = 1
            else:
                newstates[i] = 0
                newstates_mask[i] = 0


        future_rewards = self.s.run(self.future_rewards, {
            self.observation:      newstates,
            self.observation_mask: newstates_mask,
            self.rewards:          rewards,
        })

        res = self.s.run([self.prediction_error, self.train_op] + self.metrics, {
            self.observation:                states,
            self.action_mask:                action_mask,
            self.precomputed_future_rewards: future_rewards,
        })
        cost, metrics = res[0], res[2:]

        if self.summary_writer is not None:
            for metric in metrics:
                self.summary_writer.add_summary(metric, self.iteration)
        self.iteration += 1
