import gym

from baselines import deepq
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=2000000)
parser.add_argument('--prior', type=str, default=None) # models/pong_fully_trained_2.pkl"
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--k', type=float, default=None)
args = parser.parse_args()

def main():
    env = gym.make("PongNoFrameskip-v4")
    env = ScaledFloatFrame(wrap_dqn(env))

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.iter,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=False,
        score_limit=None,
        scope="deepq",
        prior_fname=args.prior,
        model_name = args.name,
        softq_k=args.k
    )
    print("FINISHED.")
    act.save("models/{}.pkl".format())
    env.close()


if __name__ == '__main__':
    main()
