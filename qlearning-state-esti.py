import time

from battle_field import BattleField,FieldWrapper
from agent import QLearningTacitAgent


def test_episode(env, agents):
    total_reward = 0
    obs = env.reset()
    env.render()

    # unify the inner states of agents
    for idx in range(env.n_agents):
        agents[idx].unify(obs)

    while True:
        actions=[]
        for idx in range(env.n_agents):
            action=agents[idx].predict(obs[0])[idx]     # give only the observation of the elephant to the agent
            actions.append(action)
        next_obs, reward, done, _ = env.step(actions)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break
    return total_reward


def main():
    field_size=4
    N_preyers=3
    ele_goal=(1,2)
    episode_limit=15
    env = BattleField(field_size, N_preyers, ele_goal, episode_limit)
    env = FieldWrapper(env)

    agents=[]
    preyer1=QLearningTacitAgent(
        obs_n=env.n_obss,
        act_n=env.n_actions,
        learning_rate=0.1,
        gamma=0.8,
        e_greed=0.2)
    preyer1.restore()
    agents.append(preyer1)

    preyer2=QLearningTacitAgent(
        obs_n=env.n_obss,
        act_n=env.n_actions,
        learning_rate=0.1,
        gamma=0.8,
        e_greed=0.2)
    preyer2.restore()
    agents.append(preyer2)

    preyer3=QLearningTacitAgent(
        obs_n=env.n_obss,
        act_n=env.n_actions,
        learning_rate=0.1,
        gamma=0.8,
        e_greed=0.2)
    preyer3.restore()
    agents.append(preyer3)


    # num_of_test
    num_of_test = 10
    total_reward = 0
    for _ in range(num_of_test):
        reward=test_episode(env, agents)
        total_reward += reward
    print('average reward = %.1f' % (total_reward / num_of_test))


    # # 各航行器同化 synch
    # preyer1.synch(obs,action_ele,'explicit')
    # preyer2.synch(obs,action_ele,'explicit')
    # preyer3.synch(obs,action_ele,'explicit')
    # # preyer1.synch(obs,action_ele,'tacit')
    # # preyer2.synch(obs,action_ele,'tacit')
    # # preyer3.synch(obs,action_ele,'tacit')


if __name__=='__main__':
    main()