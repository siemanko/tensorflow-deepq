import numpy as np
import random
import tensorflow as tf

from collections import deque

class ContinuousDeepQ(object):
    def __init__(self, observation_size,
                       action_size,
                       actor,
                       critic,
                       optimizer,
                       session,
                       exploration_sigma=0.05,
                       exploration_period=1000,
                       store_every_nth=5,
                       train_every_nth=5,
                       minibatch_size=32,
                       discount_rate=0.95,
                       max_experience=30000,
                       target_actor_update_rate=0.01,
                       target_critic_update_rate=0.01,
                       summary_writer=None):
        """Initialized the Deepq object.

        Based on:
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        Parameters
        -------
        observation_size : int
            length of the vector passed as observation
        action_size : int
            length of the vector representing an action
        observation_to_actions: dali model
            model that implements activate function
            that can take in observation vector or a batch
            and returns scores (of unbounded values) for each
            action for each observation.
            input shape:  [batch_size, observation_size]
            output shape: [batch_size, action_size]
        optimizer: tf.solver.*
            optimizer for prediction error
        session: tf.Session
            session on which to execute the computation
        exploration_sigma: float (0 to 1)
        exploration_period: int
            probability of choosing a random
            action (epsilon form paper) annealed linearly
            from 1 to exploration_sigma over
            exploration_period
        store_every_nth: int
            to further decorrelate samples do not all
            transitions, but rather every nth transition.
            For example if store_every_nth is 5, then
            only 20% of all the transitions is stored.
        train_every_nth: int
            normally training_step is invoked every
            time action is executed. Depending on the
            setup that might be too often. When this
            variable is set set to n, then only every
            n-th time training_step is called will
            the training procedure actually be executed.
        minibatch_size: int
            number of state,action,reward,newstate
            tuples considered during experience reply
        dicount_rate: float (0 to 1)
            how much we care about future rewards.
        max_experience: int
            maximum size of the reply buffer
        target_actor_update_rate: float
            how much to update target critci after each
            iteration. Let's call target_critic_update_rate
            alpha, target network T, and network N. Every
            time N gets updated we execute:
                T = (1-alpha)*T + alpha*N
        target_critic_update_rate: float
            analogous to target_actor_update_rate, but for
            target_critic
        summary_writer: tf.train.SummaryWriter
            writer to log metrics
        """
        # memorize arguments
        self.observation_size          = observation_size
        self.action_size               = action_size

        self.actor                     = actor
        self.critic                    = critic
        self.optimizer                 = optimizer
        self.s                         = session

        self.exploration_sigma         = exploration_sigma
        self.exploration_period        = exploration_period
        self.store_every_nth           = store_every_nth
        self.train_every_nth           = train_every_nth
        self.minibatch_size            = minibatch_size
        self.discount_rate             = tf.constant(discount_rate)
        self.max_experience            = max_experience

        self.target_actor_update_rate = \
                tf.constant(target_actor_update_rate)
        self.target_critic_update_rate = \
                tf.constant(target_critic_update_rate)

        # deepq state
        self.actions_executed_so_far = 0
        self.experience = deque()

        self.iteration = 0
        self.summary_writer = summary_writer

        self.number_of_times_store_called = 0
        self.number_of_times_train_called = 0

        self.create_variables()

    @staticmethod
    def linear_annealing(n, total, p_initial, p_final):
        """Linear annealing between p_initial and p_final
        over total steps - computes value at step n"""
        if n >= total:
            return p_final
        else:
            return p_initial - (n * (p_initial - p_final)) / (total)

    @staticmethod
    def update_target_network(source_network, target_network, update_rate):
        target_network_update = []
        for v_source, v_target in zip(source_network.variables(), target_network.variables()):
            # this is equivalent to target = (1-alpha) * target + alpha * source
            update_op = v_target.assign_sub(update_rate * (v_target - v_source))
            target_network_update.append(update_op)
        return tf.group(*target_network_update)

    def create_variables(self):
        self.target_actor  = self.actor.copy(scope="target_actor")
        self.target_critic = self.critic.copy(scope="target_critic")

        # FOR REGULAR ACTION SCORE COMPUTATION
        with tf.name_scope("taking_action"):
            self.observation  = tf.placeholder(tf.float32, (None, self.observation_size), name="observation")
            self.actor_action = tf.identity(self.actor(self.observation), name="actor_action")
            tf.histogram_summary("actions", self.actor_action)

        # FOR PREDICTING TARGET FUTURE REWARDS
        with tf.name_scope("estimating_future_reward"):
            self.next_observation          = tf.placeholder(tf.float32, (None, self.observation_size), name="next_observation")
            self.next_observation_mask     = tf.placeholder(tf.float32, (None,), name="next_observation_mask")
            self.next_action               = self.target_actor(self.next_observation) # ST
            tf.histogram_summary("target_actions", self.next_action)
            self.next_value                = self.target_critic([self.next_observation, self.next_action]) # ST
            self.rewards                   = tf.placeholder(tf.float32, (None,), name="rewards")
            self.future_reward             = self.rewards + self.discount_rate *  self.next_observation_mask * self.next_value

        with tf.name_scope("critic_update"):
            ##### ERROR FUNCTION #####
            self.given_action               = tf.placeholder(tf.float32, (None, self.action_size), name="given_action")
            self.value_given_action         = self.critic([self.observation, self.given_action])
            tf.scalar_summary("value_for_given_action", tf.reduce_mean(self.value_given_action))
            temp_diff                       = self.value_given_action - self.future_reward

            self.critic_error               = tf.reduce_mean(tf.square(temp_diff))
            ##### OPTIMIZATION #####
            critic_gradients                       = self.optimizer.compute_gradients(self.critic_error, var_list=self.critic.variables())
            # Add histograms for gradients.
            for grad, var in critic_gradients:
                tf.histogram_summary('critic_update/' + var.name, var)
                if grad:
                    tf.histogram_summary('critic_update/' + var.name + '/gradients', grad)
            self.critic_update              = self.optimizer.apply_gradients(critic_gradients)
            tf.scalar_summary("critic_error", self.critic_error)

        with tf.name_scope("actor_update"):
            ##### ERROR FUNCTION #####
            self.actor_score = self.critic([self.observation, self.actor_action])

            ##### OPTIMIZATION #####
            # here we are maximizing actor score.
            # only optimize actor variables here, while keeping critic constant
            actor_gradients = self.optimizer.compute_gradients(tf.reduce_mean(-self.actor_score), var_list=self.actor.variables())
            # Add histograms for gradients.
            for grad, var in actor_gradients:
                tf.histogram_summary('actor_update/' + var.name, var)
                if grad:
                    tf.histogram_summary('actor_update/' + var.name + '/gradients', grad)
            self.actor_update              = self.optimizer.apply_gradients(actor_gradients)
            tf.scalar_summary("actor_score", tf.reduce_mean(self.actor_score))

        # UPDATE TARGET NETWORK
        with tf.name_scope("target_network_update"):
            self.target_actor_update  = ContinuousDeepQ.update_target_network(self.actor, self.target_actor, self.target_actor_update_rate)
            self.target_critic_update = ContinuousDeepQ.update_target_network(self.critic, self.target_critic, self.target_critic_update_rate)
            self.update_all_targets = tf.group(self.target_actor_update, self.target_critic_update)

        self.summarize = tf.merge_all_summaries()
        self.no_op1 = tf.no_op()

    def action(self, observation, disable_exploration=False):
        """Given observation returns the action that should be chosen using
        DeepQ learning strategy. Does not backprop."""
        assert len(observation.shape) == 1, \
                "Action is performed based on single observation."

        self.actions_executed_so_far += 1
        noise_sigma = ContinuousDeepQ.linear_annealing(self.actions_executed_so_far,
                                                       self.exploration_period,
                                                       1.0,
                                                       self.exploration_sigma)

        action = self.s.run(self.actor_action, {self.observation: observation[np.newaxis,:]})[0]
        if not disable_exploration:
            action += np.random.normal(0, noise_sigma, size=action.shape)
            action = np.clip(action, -1., 1.)

        return action

    def store(self, observation, action, reward, newobservation):
        """Store experience, where starting with observation and
        execution action, we arrived at the newobservation and got thetarget_network_update
        reward reward

        If newstate is None, the state/action pair is assumed to be terminal
        """
        if self.number_of_times_store_called % self.store_every_nth == 0:
            self.experience.append((observation, action, reward, newobservation))
            if len(self.experience) > self.max_experience:
                self.experience.popleft()
        self.number_of_times_store_called += 1

    def training_step(self):
        """Pick a self.minibatch_size exeperiences from reply buffer
        and backpropage the value function.
        """
        if self.number_of_times_train_called % self.train_every_nth == 0:
            if len(self.experience) <  self.minibatch_size:
                return

            # sample experience (need to be a twoliner, because deque...)
            samples   = random.sample(range(len(self.experience)), self.minibatch_size)
            samples   = [self.experience[i] for i in samples]

            # bach states
            states         = np.empty((len(samples), self.observation_size))
            newstates      = np.empty((len(samples), self.observation_size))
            actions        = np.zeros((len(samples), self.action_size))

            newstates_mask = np.empty((len(samples),))
            rewards        = np.empty((len(samples),))

            for i, (state, action, reward, newstate) in enumerate(samples):
                states[i] = state
                actions[i] = action
                rewards[i] = reward
                if newstate is not None:
                    newstates[i] = newstate
                    newstates_mask[i] = 1.
                else:
                    newstates[i] = 0
                    newstates_mask[i] = 0.

            _, _, summary_str = self.s.run([
                self.actor_update,
                self.critic_update,
                self.summarize if self.iteration % 100 == 0 else self.no_op1,
            ], {
                self.observation:            states,
                self.next_observation:       newstates,
                self.next_observation_mask:  newstates_mask,
                self.given_action:           actions,
                self.rewards:                rewards,
            })

            self.s.run(self.update_all_targets)

            if self.summary_writer is not None and summary_str is not None:
                self.summary_writer.add_summary(summary_str, self.iteration)

            self.iteration += 1

        self.number_of_times_train_called += 1
