import time

from battle_field import BattleField,FieldWrapper
from agent import QLearningTacitAgent
from utils import to_points,acts2num

def test_episode(env, agents):
    step=0
    total_reward = 0
    obs = env.reset()
    env.render()
    unify_flag=True

    while True:
        # Initialize the inner states of agents at the beginning of each episode
        if unify_flag:
            print(f'Unify at step {step}')
            # unify the inner states of agents
            for idx in range(env.N_preyers):
                agents[idx].unify(obs)
            unify_flag=False

        # make the partial obs
        partial_obs=to_points(obs,env.N_preyers+1,env.field_size,16)[0]
        # sample actions and step
        actions=[]
        for idx in range(env.N_preyers):
            predicted_actions,conflict=agents[idx].sample(partial_obs)     # give only the observation of the elephant to the agent
            actions.append(predicted_actions[idx])
            if conflict:
                print(f'Conflict at step {step}')
                unify_flag=True
                break

        # actions=[0,0,0], the action of the preyers here is the real action, not the xcoded number
        action=acts2num(actions,xcimal=5)
        next_obs, reward, done, _ = env.step(action)
        step+=1

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
        field_size=field_size,
        N_preyers=N_preyers,
        ele_goal=ele_goal,
        learning_rate=0.1,
        gamma=0.8,
        e_greed=0.2)
    preyer1.restore()
    agents.append(preyer1)

    preyer2=QLearningTacitAgent(
        obs_n=env.n_obss,
        act_n=env.n_actions,
        field_size=field_size,
        N_preyers=N_preyers,
        ele_goal=ele_goal,
        learning_rate=0.1,
        gamma=0.8,
        e_greed=0.2)
    preyer2.restore()
    agents.append(preyer2)

    preyer3=QLearningTacitAgent(
        obs_n=env.n_obss,
        act_n=env.n_actions,
        field_size=field_size,
        N_preyers=N_preyers,
        ele_goal=ele_goal,
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


if __name__=='__main__':
    main()