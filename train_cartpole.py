import gym
import argparse
from baselines import deepq
import time

def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    time_of_training = time.strftime("%Y-%m-%dT%H:%M:%S")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #softq, doubleq, nodoubleq, priortrain
    parser.add_argument('--mode', type=str, required=True)

    # linear, constant
    parser.add_argument('--beta-schedule', type=str, default=None)

    # if constant, then beta = beta-param. if linear, then beta = beta-param * num-steps
    parser.add_argument('--beta-param', type=float, default=None)

===    parser.add_argument('--prior', type=str, default=None)
===    parser.add_argument('--num-runs', type=int, default=1)
===    parser.add_argument('--score-limit', type=int, default=None)
===    parser.add_argument('--logdir', type=str, default="cartpole_results/default_saved/{}".format(time_of_training))
===    parser.add_argument('--model-dir', type=str, default=None)

    args = parser.parse_args()
    assert mode == "doubleq" or mode == "priortrain" or mode == "nodoubleq"

    scope = "deepq"
    if args.mode == "priortrain":
        assert args.model_dir is not None
        scope = "prior"


    prior = None
    if args.prior:
        prior = deepq.load(args.prior, scope="prior")


    for i in range(args.num_runs):
        env = gym.make("CartPole-v0")
        model = deepq.models.mlp([64])

        act = deepq.learn(
            env,
            q_func=model,
            logdir=args.logdir,
            mode=args.mode,
            lr=1e-3,
            max_timesteps=100000,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            print_freq=1,
            callback=callback,
            prioritized_replay=True,
            score_limit=args.score_limit,
            scope=scope,
            prior=prior
        )
        print("Saving model")
        if args.model_dir is not None:
            act.save(args.model_dir)



if __name__ == '__main__':
    main()
