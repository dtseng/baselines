import gym

from baselines import deepq


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make("CartPole-v0")
    model = deepq.models.mlp([64])

    prior = deepq.load("models/cartpole_fully_trained.pkl", scope="prior")

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=1,
        callback=callback,
        prioritized_replay=True,
        score_limit=200,
        scope="deepq",
        prior=prior
    )
    # print("Saving model to cartpole_model.pkl")


if __name__ == '__main__':
    main()
