import gym
import sys
from baselines import deepq
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame


def main():
    env = gym.make("PongNoFrameskip-v4")
    env = ScaledFloatFrame(wrap_dqn(env))

    prior_fname = "models/pong_fully_trained_2.pkl"

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=2000000,
        buffer_size=10000,
        exploration_fraction=sys.argv[4],
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        score_limit=None,
        scope="deepq",
        prior_fname=prior_fname
    )
    print("FINISHED.")
    # act.save("models/soft_q_with_strong_prior.pkl")
    env.close()


if __name__ == '__main__':
    main()
