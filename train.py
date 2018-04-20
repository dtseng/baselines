import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile, zipfile, dill
import time


import baselines.common.tf_util as U
import math
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
# when updating this to non-deperecated ones, it is important to
# copy over LazyFrames
from baselines.common.atari_wrappers_deprecated import wrap_dqn
# from baselines.common.azure_utils import Container
from baselines.deepq.experiments.atari.model import model, dueling_model, mlp_model


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @classmethod
    def load(self, path, sess_exists, num_cpu=16, scope="deepq"):
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

def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Pong", help="name of the game")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=float, default=5e7, help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=float, default=1e4, help="number of iterations between every target network update")
    # Bells and whistles
    boolean_flag(parser, "double-q", default=False, help="whether or not to use double q learning")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "prioritized", default=False, help="whether or not to use prioritized replay buffer")
    parser.add_argument("--prioritized-alpha", type=float, default=0.6, help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4, help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
    # Checkpointing
    parser.add_argument("--save-loc", type=str, default=None, help="directory in which model should be saved.")
    parser.add_argument("--prior-loc", type=str, default=None, help="directory in which prior model is located.")

    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=float, default=1e6, help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
    parser.add_argument("--log-dir", type=str, default=None, help="directory in which logs should be saved.")
    boolean_flag(parser, "softq", default=False, help="whether or not to use softq learning")
    parser.add_argument("--k", type=float, default=1e-6, help="scheduling for beta")
    parser.add_argument("--softmax-weight", type=float, default=1, help="scheduling for beta")

    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
    env = wrap_dqn(monitored_env)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)

    return env, monitored_env


# def maybe_save_model(savedir, container, state):
#     """This function checkpoints the model and state of the training algorithm."""
#     if savedir is None:
#         return
#     start_time = time.time()
#     model_dir = "model-{}".format(state["num_iters"])
#     U.save_state(os.path.join(savedir, model_dir, "saved"))
#     if container is not None:
#         container.put(os.path.join(savedir, model_dir), model_dir)
#     relatively_safe_pickle_dump(state, os.path.join(savedir, 'training_state.pkl.zip'), compression=True)
#     if container is not None:
#         container.put(os.path.join(savedir, 'training_state.pkl.zip'), 'training_state.pkl.zip')
#     relatively_safe_pickle_dump(state["monitor_state"], os.path.join(savedir, 'monitor_state.pkl'))
#     if container is not None:
#         container.put(os.path.join(savedir, 'monitor_state.pkl'), 'monitor_state.pkl')
#     logger.log("Saved model in {} seconds\n".format(time.time() - start_time))
#

# def maybe_load_model(savedir, container):
#
#     """Load model if present at the specified path."""
#     if savedir is None:
#         return
#
#     state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
#     print("got state path....")
#     if container is not None:
#         logger.log("Attempting to download model from Azure")
#         found_model = container.get(savedir, 'training_state.pkl.zip')
#     else:
#         found_model = os.path.exists(state_path)
#     print("found model...?")
#     if found_model:
#         state = pickle_load(state_path, compression=True)
#         print("loaded state...")
#         model_dir = "model-{}".format(state["num_iters"])
#         if container is not None:
#             container.get(savedir, model_dir)
#         U.load_state(os.path.join(savedir, model_dir, "saved"))
#         print("loaded model...")
#         logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
#         return state


def test_rollout(num_episodes, env, act):
    for _ in range(num_episodes):
        obs, done = env.reset(), False
        while not done:
            obs, rew, done, info = env.step(act(np.array(obs)[None])[2][0])

    return np.mean(info["rewards"][-num_episodes:])


if __name__ == '__main__':
    args = parse_args()

    # Create and seed the env.
    env, monitored_env = make_env(args.env)
    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)

    def make_obs_ph(name):
        return U.Uint8Input(env.observation_space.shape, name=name)

    q_func = dueling_model if args.dueling else model


    with U.make_session(4) as sess:
        summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph, flush_secs=10)

        use_prior = args.prior_loc is not None
        # Create training graph and replay buffer
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=env.action_space.n,
            # optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            optimizer=tf.train.RMSPropOptimizer(learning_rate=2.5e-4, epsilon=0.01, decay=0.95, momentum=0.95),
            args=args,
            gamma=0.99,
            grad_norm_clipping=10,
            double_q=args.double_q,
            scope="softq",
            use_prior=use_prior,
        )

        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func,
            'num_actions': env.action_space.n,
        }

        approximate_num_iters = args.num_steps / 4
        exploration = PiecewiseSchedule([
            (0, 1.0),
            (5e4, 1.0),
            (5e4+1e6, 0.1)
        ], outside_value=0.1)

        if args.prioritized:
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(approximate_num_iters, initial_p=args.prioritized_beta0, final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(args.replay_buffer_size)

        U.initialize()
        update_target()

        num_iters = 0

        if args.prior_loc:
            prior = ActWrapper.load(args.prior_loc, sess_exists=True, scope="deepq")

        start_time, start_steps = None, None
        steps_per_iter = RunningAvg(0.999)
        iteration_time_est = RunningAvg(0.999)
        obs = env.reset()

        saved_mean_reward = None
        mean_100ep_reward = None

        # Main training loop
        while True:
            num_iters += 1
            # Take action and store transition in the replay buffer.
            action = act(np.array(obs)[None], update_eps=exploration.value(num_iters))[0][0]
            new_obs, rew, done, info = env.step(action)

            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs
            if done:
                obs = env.reset()

            if num_iters > 5e4 and num_iters % 4 == 0:
                # Sample a bunch of transitions from replay buffer
                if args.prioritized:
                    experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                    weights = np.ones_like(rewards)

                if use_prior:
                    prior_policy = softmax(args.softmax_weight*prior(obses_tp1)[1])
                else:
                    prior_policy = np.zeros((obses_t.shape[0], env.action_space.n))
                # Minimize the error in Bellman's equation and compute TD-error
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights, np.int64(info["steps"]), prior_policy)
                # Update the priorities in the replay buffer
                if args.prioritized:
                    new_priorities = np.abs(td_errors) + args.prioritized_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
            # Update target network.
            if num_iters % args.target_update_freq == 0:
                update_target()

            if start_time is not None:
                steps_per_iter.update(info['steps'] - start_steps)
                iteration_time_est.update(time.time() - start_time)

            start_time, start_steps = time.time(), info["steps"]

            if info["steps"] > args.num_steps:
                break

            if done:
                steps_left = args.num_steps - info["steps"]
                completion = np.round(info["steps"] / args.num_steps, 1)
                mean_100ep_reward = np.mean(info["rewards"][-100:])

                logger.record_tabular("% completion", completion)
                logger.record_tabular("steps", info["steps"])
                logger.record_tabular("iters", num_iters)
                logger.record_tabular("episodes", len(info["rewards"]))

                logger.record_tabular("reward (100 epi mean)", mean_100ep_reward)
                logger.record_tabular("exploration", exploration.value(num_iters))

                if len(info["rewards"]) > 0:
                    summary = tf.Summary()
                    summary.value.add(tag='return', simple_value=info["rewards"][-1])
                    summary.value.add(tag='mean100_return', simple_value=mean_100ep_reward)
                    summary.value.add(tag='beta', simple_value=info["steps"]*args.k)
                    summary.value.add(tag='exploration', simple_value=exploration.value(num_iters))

                    # if len(info["rewards"]) % 50 == 0: #!! Need to fix. This uses info['steps'], which I think screws things up
                    #     mean_rollout_reward = test_rollout(5, env, act)
                    #     summary.value.add(tag='rollout_reward', simple_value=mean_rollout_reward)
                    #     obs = env.reset()

                    summary_writer.add_summary(summary, num_iters)

                if args.prioritized:
                    logger.record_tabular("max priority", replay_buffer._max_priority)
                fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else "calculating...")
                logger.dump_tabular()
                logger.log()
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()


            if args.save_loc and args.save_freq and mean_100ep_reward is not None and num_iters > 0 and (num_iters % args.save_freq == 0 or info["steps"] > args.num_steps):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    ActWrapper(act, act_params).save(args.save_loc)
                    saved_mean_reward = mean_100ep_reward
