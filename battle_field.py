import random

import numpy as np
from scipy.spatial import distance


from env_base import Env
from elephant import Elephant
from preyers import Preyer


class Battlefield(Env):
    def __init__(self,field_size,N_preyers,ele_goal,episode_limit=15):
        super(Battlefield,self).__init__()
        self.step_count=0
        self.episode_limit=episode_limit
        # create a battle field by size and put elephant and preyers
        self.field_size=field_size
        self.field=np.zeros((field_size,field_size))
        self.N_preyers=N_preyers

        # deafault goal of the elephant, set a input variable if need
        self.ele_goal=ele_goal
        self.ele_agent=Elephant(ele_goal,obs_range=1,style='aggressive')      # the style can be choose from 'aggressive', 'moderate' and 'conservative'

        # added for qmix
        self.n_actions=4
        self.win_counted=0
        self.last_obs=None
        self.last_action=[-1,-1,-1]
        self.obss=None
        
    # added for qmix
    def get_env_info(self):
        env_info={
            'episode_limit':self.episode_limit,
            'n_actions':self.n_actions,
            'n_agents':self.N_preyers,
            'state_shape':63,
            'obs_shape':12,
        }
        return env_info
        
    # added for qmix
    def get_obs(self):
        obs=[]
        ele_pos=self.ele_pos
        for idx,preyer_pos in self.preyers.items():
            rel_pos=np.array(preyer_pos)-np.array(ele_pos)
            dis=np.linalg.norm(rel_pos)
            rel_pos_scale=(rel_pos+np.array([1,1]))/self.field_size
            dis_scale=dis/self.field_size
            obs.append([rel_pos_scale[0],rel_pos_scale[1],dis_scale])
        

            for idx0,preyer_pos0 in self.preyers.items():
                if idx==idx0:
                    obs[idx-1]=obs[idx-1]+[0,0,0]
                else:
                    rel_pos=np.array(preyer_pos)-np.array(preyer_pos0)
                    dis=np.linalg.norm(rel_pos)
                    rel_pos_scale=(rel_pos+np.array([1,1]))/self.field_size
                    dis_scale=dis/self.field_size
                    obs[idx-1]=obs[idx-1]+[rel_pos_scale[0],rel_pos_scale[1],dis_scale]

        self.obss=obs
        return obs

    # added for qmix
    def get_state(self):
        field_center=np.array((self.field_size-1,self.field_size-1))/2
        rel_ele_pos=np.array(self.ele_pos)-field_center
        rel_margin_ele_pos=np.array(self.ele_pos)
        rel_margin_ele_pos_neg=np.array((self.field_size-1,self.field_size-1))-np.array(self.ele_pos)
        state=np.hstack((rel_ele_pos,rel_margin_ele_pos,rel_margin_ele_pos_neg))
        for idx, preyer_pos in self.preyers.items():
            rel_preyer_pos=np.array(preyer_pos)-field_center
            rel_margin_preyer_pos=np.array(preyer_pos)
            rel_margin_preyer_pos_neg=np.array((self.field_size-1,self.field_size-1))-np.array(preyer_pos)
            state=np.hstack((state,rel_preyer_pos,rel_margin_preyer_pos,rel_margin_preyer_pos_neg))
        # scale
        state/=self.field_size
        # add obss
        obss_state=np.reshape(self.obss,12*self.N_preyers,)
        action_state=np.array(self.last_action)/self.n_actions
        state=np.hstack((state,obss_state,action_state))

        # field_state=np.reshape(self.field,self.field_size**2)
        # obss_state=np.reshape(self.obss,3*self.N_preyers,)
        # state=np.hstack((field_state,obss_state))
        return state

    def calc_reward(self,done,info):
        # ele_pos=self.ele_pos
        # ele_goal=self.ele_goal
        # reward=1/2/5-1/np.linalg.norm(np.array(ele_pos)-np.array(ele_goal))
        reward=0
        if not done:
            return reward
        else:
            if 'Surrounded!' in info:
                reward+=100
            if 'Ele Wins!' in info:
                reward-=100
            if 'Crush!' in info:
                reward-=50
            
            return reward

    def _update_field(self):
        # update the position of the ele and the preyers
        field_size=self.field_size
        field=np.zeros((field_size,field_size))
        field[self.ele_pos]=-1     # '-1' stands for the elephant
        for idx,pos in self.preyers.items():
            field[pos]=idx     # use the 'idx's to distinguish the preyers
        return field
    
    def reset(self):
        self.step_count=0
        # relocate the elephant and the preyers
        self.ele_pos=(0,0)      # every time from top-left to right_down
        # added for qmix
        self.win_counted=0

        field_size=self.field_size
        N_preyers=self.N_preyers
        self.preyers={}
        preyer_set=set()
        preyer_set.add(self.ele_pos)
        while len(preyer_set)<N_preyers+1:
            preyer_pos=(
                random.randint(0,field_size-1),
                random.randint(0,field_size-1),
            )
            if preyer_pos not in preyer_set:
                self.preyers[len(preyer_set)]=preyer_pos
                preyer_set.add(preyer_pos)
        field=self._update_field()
        # self.field_old=self.field=field   # reserve here for the field_old
        self.field=field
        obs={
            'field':field,
            'ele_pos':self.ele_pos,
            'preyers':self.preyers,
        }
        self.last_obs=obs
        return obs

    def if_done(self,status):
        done=False
        info_list=[]
        if self._if_surrounded():
            done=True
            self.win_counted+=1
            info_list.append('Surrounded!')
        if self._if_time_out():
            done=True
            info_list.append('Time out!')
        if self._if_ele_wins():
            done=True
            info_list.append('Ele Wins!')
        if status=='crashed':
            done=True
            info_list.append('Crush!')
        if done:
            return True,info_list
        else:
            return False,[]


    def _if_surrounded(self):
        surrounded_list=[]
        for _, pos in self.preyers.items():
            if distance.cityblock(self.ele_pos,pos)==1:
                surrounded_list.append(pos)
        N_preyers=self.N_preyers
            
        if not self._at_corner():
            # when the elephant is not at the corner
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
        else:     
            # the elephant at the corner
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

    def step(self,action): 
        self.field,status =self._update_pos(action)
        self.last_action=action
        self.step_count+=1
        obs={
            'field':self.field,
            'ele_pos':self.ele_pos,
            'preyers':self.preyers
        }
        self.last_obs=obs
        
        done,info=self.if_done(status)
        reward=self.calc_reward(done,info)
        return reward,done,info

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

    def _update_pos(self,action):
        status='normal'
        # update the pose of the elephant
        act_ele=self.ele_agent.sample(self.last_obs)
        pos_set=set()
        pos_new=self.get_x_y(self.ele_pos,act_ele)
        self.ele_pos=pos_new
        pos_set.add(pos_new)

        # update the poses of the preyers
        # some actions can't be applied for the positions may be occupied by the elephant.
        for idx,pos in self.preyers.items():
            act=action[idx-1]
            # if len(action[idx])==1:
            #     act=action[idx]     # let us define the action by a python dict
            # elif len(action[idx])==3:
            #     act=action[idx][idx-1]
            pos_new=self.get_x_y(pos,act)
            if pos_new not in pos_set:
                pos_set.add(pos_new)
                self.preyers[idx]=pos_new
            elif pos not in pos_set:
                pos_set.add(pos)
                self.preyers[idx]=pos
            else:
                status='crashed'
                print('ERROR! Preyer Crush!')

        field=np.zeros_like(self.field)
        field[self.ele_pos]=-1     # '-1' stands for the elephant
        for idx,pos in self.preyers.items():
            field[pos]=idx     # use the 'idx's to distinguish the preyers
        return field, status

    def print_field(self):
        print(self.field)


    def get_avail_agent_actions(self,agent_id):
        preyer_pos=self.preyers[agent_id+1]
        avail_agent_actions=np.zeros(self.n_actions)
        for idx, act in enumerate([2,4,6,8]):
            pos_new=self.get_x_y(preyer_pos,act)
            if pos_new!=preyer_pos:
                avail_agent_actions[idx]=1
        return avail_agent_actions      # return a one-hot form action




if __name__ == '__main__':
    random.seed(123)
    goal=(8,8)
    env=Battlefield(9,3,goal)
    obs=env.reset()
    ele_agent=Elephant(goal,style='conservative')      # the style can be choose from 'aggressive', 'moderate' and 'conservative'

    preyer1=Preyer('P1',1,obs)
    preyer2=Preyer('P2',2,obs)
    preyer3=Preyer('P3',3,obs)
    for i in range(10):
        print("=============================================")
        env.print_field()
        action1=preyer1.sample(obs)
        action2=preyer2.sample(obs)
        action3=preyer3.sample(obs)
        action={-1:int(ele_agent.sample(obs)),1:action1,2:action2,3:action3}
        obs,reward,done,info=env.step(action)
        if done==True:
            result_info=''
            for res_info in info:
                result_info+=res_info
            print(f'done and the result is {result_info}')
            env.print_field()
            break
    
    