from battle_field import BattleField
from agent import QLearningAgent
import time

def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    env.render()
    while True:
        avail_actions=env.get_avail_agent_actions()
        action = agent.predict(obs,avail_actions)  # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break


def main():
    field_size=4
    N_preyers=3
    ele_goal=(1,2)
    episode_limit=15
    env = BattleField(field_size, N_preyers, ele_goal, episode_limit)

    agent = QLearningAgent(
        obs_n=env.n_obss,
        act_n=env.n_actions,
        learning_rate=0.1,
        gamma=0.8,
        e_greed=0.2)
    agent.restore()
    for _ in range(5):
        test_episode(env, agent)


if __name__ == "__main__":
    main()