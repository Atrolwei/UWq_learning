from agents_base import Agent
from scipy.spatial import distance
import numpy as np 
from math import atan2,pi


class Elephant(Agent):
    def __init__(self,goal,field_size,obs_range=5,style='moderate'):
        super(Elephant,self).__init__()
        self.goal=goal
        self.field_size=field_size
        # define the observation range of the agent
        self.obs_range=obs_range
        # give the agent a style therefore a changeable strategy
        if style=='aggressive':
            self.K_go=0.25
        elif style=='moderate':
            self.K_go=0.5
        elif style=='conservative':
            self.K_go=1


    def sample(self,obs):
        force=self._potential_field(obs)
        action=self._get_action_by_force(force)
        return action

    def _potential_field(self,obs):
        obs_range=self.obs_range

        # the force of the potential field
        force=np.array([0.0,0.0])

        # intrinsive force of the goal
        ele_pos=obs[0]
        delta_goal=np.array(self.goal)-np.array(ele_pos)
        delta_goal_norm=np.linalg.norm(delta_goal)
        if delta_goal_norm:
            force+=1/delta_goal_norm*delta_goal/delta_goal_norm
    
        # resistance forces
        delta_force=np.array([0.0,0.0])
        K_go=self.K_go
        N_preyer_obs=0

        for preyer_pos in obs[1:]:
            if distance.euclidean(ele_pos,preyer_pos)<=obs_range:
                # print(f'Preyer found, index: {idx}, position: {preyer_pos}')
                N_preyer_obs+=1     # count the number of the preyers observed
                delta_preyer=np.array(preyer_pos)-np.array(ele_pos)
                delta_preyer_norm=np.linalg.norm(delta_preyer)
                if not delta_preyer_norm:
                    print(f'delta_preyer_norm is 0, preyer_pos: {preyer_pos}, ele_pos: {ele_pos}')
                delta_force-=K_go*(1/delta_preyer_norm-1/obs_range)*delta_preyer/delta_preyer_norm
        if N_preyer_obs>0:
            delta_force/=N_preyer_obs
        force+=delta_force      # the force is a 2 dim vector means the force induced by the goal and the preyers
        return force

    def _get_action_by_force(self,force):
        # decide the action by the force
        action=5
        force_norm=np.linalg.norm(force)
        if force_norm<0.02:
            action=5
        else:
            theta=atan2(-force[1],force[0])
            theta_deg_round=round(theta/pi*180)
            if theta_deg_round<0:
                theta_deg_round+=360
            n=theta_deg_round//45
            mod=theta_deg_round%45
            n_mod=round(mod/45)
            n=n+n_mod
            if n==0 or n==8:
                action=6
            elif n==1:
                action=9
            elif n==2:
                action=8
            elif n==3:
                action=7
            elif n==4:
                action=4
            elif n==5:
                action=1
            elif n==6:
                action=2
            elif n==7:
                action=3
        return action