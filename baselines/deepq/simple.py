import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
import zipfile
import time
import pickle
import sys

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule, ConstantSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path, sess_exists, num_cpu=16, scope="deepq"):
        with open(path, "rb") as f:
            model_data, act_params = dill.load(f)
        act = deepq.build_act(**act_params, scope=scope)
        if not sess_exists:
            sess = U.make_session(num_cpu=num_cpu)
            sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            U.load_state(os.path.join(td, "model"), scope)

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path):
        """Save model to a pickle located at `path`"""
        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            dill.dump((model_data, self._act_params), f)


def load(path, sess_exists=False, num_cpu=16, scope="deepq"):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle
    num_cpu: int
        number of cpus to use for executing the policy

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load(path, sess_exists, num_cpu=num_cpu, scope=scope)

def hard_amax(x):
    return (x == np.amax(x, axis=1, keepdims=True)).astype(int)

def softmax(x):
    y = np.amax(x, axis=1, keepdims=True)
    reduced = np.exp(x-y)
    return reduced/np.sum(reduced, axis=1, keepdims=True)

def test_rollout(num_episodes, env, act):
    episode_rew = 0
    for _ in range(num_episodes):
        obs, done = env.reset(), False
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[2][0])
            episode_rew += rew
    return episode_rew / num_episodes

def learn(env,
          q_func,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          num_cpu=16,
          score_limit=None,
          prior_fname=None,
          scope="deepq",
          rollout_period=10, # do a rollout every _ episodes
          rollout_episodes=5, # for each rollout, try _ times to get an average
          callback=None):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    num_cpu: int
        number of cpus to use for training
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = U.make_session(num_cpu=1)
    sess.__enter__()

    use_prior = prior_fname is not None

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        scope=scope,
        use_prior=use_prior
    )
    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }
    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                  initial_p=0.1,
                                  final_p=exploration_final_eps)
    #exploration = ConstantSchedule(value=0.1) # !!! Trying to mitigate covariate shift.

    summary_writer = tf.summary.FileWriter(sys.argv[1], sess.graph, flush_secs=10)

    # Initialize the parameters and copy them to the target network.
    U.initialize()

    if use_prior:
        prior = deepq.load(prior_fname, sess_exists=True, scope="prior")

    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()

    start_time = time.time()

    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        time_of_training = time.strftime("%Y-%m-%dT%H:%M:%S")
        model_file = "saved_models/{}/saved.ckpt".format(time_of_training)
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            action = act(np.array(obs)[None], update_eps=exploration.value(t))[0][0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                summary = tf.Summary()
                summary.value.add(tag='return', simple_value=episode_rewards[-1])
                summary.value.add(tag='mean100_return', simple_value=np.mean(episode_rewards[-100:]))
                summary.value.add(tag='exploration', simple_value=exploration.value(t))

                if len(episode_rewards) % rollout_period == 0:
                    mean_rollout_reward = test_rollout(rollout_episodes, env, act)
                    summary.value.add(tag='rollout_reward', simple_value=mean_rollout_reward)
                    obs = env.reset()

                summary_writer.add_summary(summary, t)

                if score_limit is not None and episode_rewards[-1] >= score_limit:
                    break
                episode_rewards.append(0.0)


            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None

                if use_prior:
                    prior_policy = hard_amax(prior(obses_tp1)[1])
                else:
                    prior_policy = np.zeros((obses_t.shape[0], env.action_space.n))
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights, t, prior_policy)

                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % 1 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("current episode reward", episode_rewards[-2])
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    U.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            # U.load_state(model_file)

    with open('saved_replay/dqn_pong.pkl', 'wb') as handle:
        pickle.dump(replay_buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return ActWrapper(act, act_params)
