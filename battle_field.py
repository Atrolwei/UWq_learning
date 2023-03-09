import random

import numpy as np
from scipy.spatial import distance


from env_base import Env
from elephant import Elephant
from preyers import Preyer
from utils import dot2num, to_xcoded_number, to_points, num2dot,num2acts


class BattleField(Env):
    def __init__(self,field_size,N_preyers,ele_goal,episode_limit=15):
        super(BattleField,self).__init__()
        self.step_count=0
        self.n_obss=16**(N_preyers+1)
        self.n_actions=4**N_preyers
        self.field_size=field_size
        self.N_preyers=N_preyers
        self.ele_goal=ele_goal        # deafault goal of the elephant, set a input variable if need
        self.episode_limit=episode_limit

        self.ele_agent=Elephant(ele_goal,field_size,obs_range=1,style='aggressive')      # the style can be choose from 'aggressive', 'moderate' and 'conservative'
        self.last_obs=None


    def reset(self):
        # randomly put the preyers and the elephant on the field
        self.step_count=0

        nums=random.sample(range(self.field_size**2),self.N_preyers+1)
        points=[]
        for num in nums:
            points.append(num2dot(num,self.field_size))
        self.ele_pos=points[0]
        self.last_ele_pos=self.ele_pos
        self.preyers_pos=points[1:]
        xcoded_number=to_xcoded_number(points,self.field_size,16)

        self.last_obs=xcoded_number

        return xcoded_number    # return the xcoded number of the initial state


    def step(self,action):
        status=self._update_pos(action)
        self.last_action=action
        self.step_count+=1
        number2be_xcoded=([self.ele_pos]+self.preyers_pos)
        obs=to_xcoded_number(number2be_xcoded,self.field_size,16)
        self.last_obs=obs
        done,info=self.if_done(status)
        reward=self.calc_reward(done,info)
        return obs,reward,done,info


    def _update_pos(self,action):
        status='normal'
        # the action of the elephant
        act_ele=self.ele_agent.sample(self.last_obs)

        pos_set=set()
        pos_new=self.get_x_y(self.ele_pos,act_ele)
        self.last_ele_pos=self.ele_pos
        self.ele_pos=pos_new
        pos_set.add(pos_new)

        # the action of the preyers
        act_preyers=num2acts(action,4)
        for idx,pos in enumerate(self.preyers_pos):
            act=[2,4,6,8][act_preyers[idx]]
            pos_new=self.get_x_y(pos,act)
            # check if the new position is valid
            if pos_new not in pos_set:
                pos_set.add(pos_new)
                self.preyers_pos[idx]=pos_new
            # if the new position is occupied by the elephant, the preyer stands still
            elif pos not in pos_set:
                pos_set.add(pos)
                self.preyers_pos[idx]=pos
            # if the new position is occupied by the other preyer, the preyer crashes
            else:
                status='crashed'
                print('ERROR! Preyer Crashes!')
                break
        return status


    def if_done(self,status):
        done=False
        info_list=[]
        if self._if_surrounded():
            done=True
            info_list.append('Surrounded!')
        if self._if_time_out():
            done=True
            info_list.append('Time out!')
        if self._if_ele_wins():
            done=True
            info_list.append('Ele Wins!')
        if status=='crashed':
            done=True
            info_list.append('Crash!')
        if done:
            return True,info_list
        else:
            return False,[]


    def _if_surrounded(self):
        surrounded_list=[]
        for pos in self.preyers_pos:
            if distance.cityblock(self.ele_pos,pos)==1:
                surrounded_list.append(pos)
        N_preyers=self.N_preyers
        
        if not self._at_corner():               # when the elephant is not at the corner
            if len(surrounded_list)==N_preyers:
                edge_len_list=[]
                for idx in range(N_preyers):
                    if idx+1<N_preyers:
                        edge_len=np.linalg.norm(
                            np.array(surrounded_list[idx])-np.array(surrounded_list[idx+1]))
                        edge_len_list.append(edge_len)
                    else:
                        edge_len=np.linalg.norm(
                            np.array(surrounded_list[idx])-np.array(surrounded_list[0]))
                        edge_len_list.append(edge_len) 
                max_len=np.array(edge_len_list).max()
                if abs(max_len-2)<1e-3 or abs(max_len-2*np.sqrt(2))<1e-3:
                    return True
                else:
                    return False
            else:
                return False
        else:                   # the elephant at the corner
            if len(surrounded_list)==N_preyers-1:
                edge_len=np.linalg.norm(
                    np.array(surrounded_list[0])-np.array(surrounded_list[1]))
                if abs(edge_len-np.sqrt(2))<1e-3:
                    return True
                else:
                    return False
            else:
                return False

    def _at_corner(self):
        ele_pos=self.ele_pos
        field_size=self.field_size
        if ele_pos in [
            (0,0),
            (field_size-1,field_size-1),
            (0,field_size-1),
            (field_size-1,0)
            ]:
            return True

    def _if_time_out(self):
        if self.step_count>10:
            return True
        else:
            return False

    def _if_ele_wins(self):
        if self.ele_pos==self.ele_goal:
            return True
        return False


    def get_x_y(self,pos,act):
        x,y=pos
        x_old,y_old=pos

        if act==1:
            y-=1
            x+=1
        elif act==2:
            x+=1
        elif act==3:
            y+=1
            x+=1
        elif act==4:
            y-=1
        elif act==5:
            pass
        elif act==6:
            y+=1
        elif act==7:
            y-=1
            x-=1
        elif act==8:
            x-=1
        elif act==9:
            y+=1
            x-=1
        else:
            pass

        size=self.field_size
        if x>size-1:
            x=size-1
            y=y_old
        elif x<0:
            x=0
            y=y_old
        if y>size-1:
            y=size-1
            x=x_old
        elif y<0:
            y=0
            x=x_old
        return x,y


    def render(self):
        field=np.zeros((self.field_size,self.field_size))
        field[self.ele_pos]=-1
        for idx,pos in enumerate(self.preyers_pos):
            field[pos]=idx+1
        print(field)
        

    def calc_reward(self,done,info):
        reward=0
        last_ele_pos=self.last_ele_pos
        ele_goal=self.ele_goal
        ele_pos=self.ele_pos
        vector0=np.array(last_ele_pos)-np.array(ele_goal)
        vector1=np.array(last_ele_pos)-np.array(ele_pos)
        num_vector=vector0.dot(vector1)
        if num_vector:
            reward=-num_vector/(np.linalg.norm(vector0)*np.linalg.norm(vector1))

        if not done:
            return reward/20    # normalize the reward
        else:
            if 'Surrounded!' in info:
                reward+=20
            if 'Ele Wins!' in info:
                reward-=20
            if 'Crash!' in info:
                reward-=10
            
            return reward/20    # normalize the reward


    def get_avail_agent_actions(self):
        avail_agent_actions=np.zeros(self.n_actions)
        for preyer_pos in self.preyers_pos:
            for act in [2,4,6,8]:
                pos_new=self.get_x_y(preyer_pos,act)
                if pos_new!=preyer_pos:
                    # avail_agent_actions[idx]=1
                    num_pos_new=dot2num(pos_new,self.field_size)
                    avail_agent_actions[num_pos_new-1]=1
        return avail_agent_actions      # return a one-hot form action




if __name__ == '__main__':
    random.seed(123)
    goal=(8,8)
    env=BattleField(9,3,goal)
    obs=env.reset()

    preyer1=Preyer('P1',1,obs)
    preyer2=Preyer('P2',2,obs)
    preyer3=Preyer('P3',3,obs)
    for i in range(10):
        print("=============================================")
        env.print_field()
        action1=preyer1.sample(obs)
        action2=preyer2.sample(obs)
        action3=preyer3.sample(obs)
        # action={-1:int(ele_agent.sample(obs)),1:action1,2:action2,3:action3}
        obs,reward,done,info=env.step(action)
        if done==True:
            result_info=''
            for res_info in info:
                result_info+=res_info
            print(f'done and the result is {result_info}')
            env.print_field()
            break
    
    