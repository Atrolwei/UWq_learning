import numpy as np


class QLearningAgent(object):
    def __init__(self,
                 obs_n,
                 act_n,
                 learning_rate=0.01,
                 gamma=0.9,
                 e_greed=0.1):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros((obs_n, act_n))

    def sample(self, obs, avail_actions):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.predict(obs, avail_actions)
        else:
            action = np.random.choice(np.where(avail_actions == 1)[0])
        return action

    def predict(self, obs, avail_actions):
        Q_list = self.Q[obs, :]
        Q_list = Q_list + avail_actions*100
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    def learn(self, obs, action, reward, next_obs, done):
        """ off-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(
                self.Q[next_obs, :])
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)

    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')


from field_state import FieldState
from utils import to_points,to_xcoded_number,num2acts


class QLearningTacitAgent(object):
    def __init__(self,
                 obs_n,
                 act_n,
                 field_size,
                 N_preyers,
                 ele_goal,
                 learning_rate=0.01,
                 gamma=0.9,
                 e_greed=0.1):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.field_state = None
        self.Q = np.zeros((obs_n, act_n))
        self.field_size=field_size
        self.N_preyers=N_preyers
        self.ele_goal=ele_goal
        self.field_state=FieldState(field_size,N_preyers,ele_goal)
        self.last_predicted_actions=None

    def unify(self,obs):    # obs here is the full obs which can be got by communication
        points=to_points(obs,self.N_preyers+1,self.field_size)
        self.field_state.reset(points)

    def sample(self,obs):
        # update the field_state and get the full obs
        full_obs,conflict=self.field_state.get_full_obs(obs)
        if conflict:
            return [5,5,5],conflict
        else:
            # get the avail_actions of the three preyers by completed full obs
            avail_actions = self.field_state.get_avail_actions()
            # predict the action of the agent
            full_obs=to_xcoded_number(full_obs,self.field_size)
            predicted_actions,done,info=self.predict(full_obs,avail_actions)
            return predicted_actions,conflict

    def predict(self, obs, avail_actions):
        Q_list = self.Q[obs, :]
        Q_list = Q_list + avail_actions*100
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = action_list[0] # actions of the 3 preyers, which is a xcoded number
        
        # use the action predicted to update the field_state
        actions=num2acts(action,self.N_preyers,5)
        actions_real=[[2,4,5,6,8][act] for act in actions]
        obs,reward,done,info = self.field_state.step(actions_real)
        return actions,done,info

    def learn(self, obs, action, reward, next_obs, done):
        """ off-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(
                self.Q[next_obs, :])
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)

    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')