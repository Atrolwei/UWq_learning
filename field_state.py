import random
import enum
import copy
import numpy as np
from scipy.spatial import distance



class Player(enum.Enum):
    ele = 1
    preyer = 2

    @property
    def other(self):
        return Player.ele if self == Player.preyer else Player.preyer

class FieldState:
    def __init__(self,obs,next_player,action,status='normal',ele_goal=(1,2)):
        # self.step_count=0
        # default goal of the elephant, set a input variable if need
        self.ele_goal=ele_goal

        # obs={
        #     'field':field,
        #     'ele_pos':self.ele_pos,
        #     'preyers':self.preyers,
        # }
        self.obs=obs
        self.field=obs['field']
        self.ele_pos=obs['ele_pos']
        self.preyers=obs['preyers']
        self.next_player = next_player
        self.last_action = action
        self.status=status

    def _update_pos(self,action):
        status='normal'
        new_field=copy.deepcopy(self.field)
        ele_pos=self.ele_pos
        preyers=self.preyers
        new_preyers=copy.deepcopy(preyers)
        if self.next_player==Player.ele:
            # update the pose of the elephant
            pos_new=self.get_x_y(ele_pos,action)
            new_field[ele_pos]=0
            new_field[pos_new]=100
            return new_field,pos_new,preyers,status

        elif self.next_player==Player.preyer: 
            pos_set=set()
            pos_set.add(ele_pos)
            # update the poses of the preyers
            # some actions can't be applied for the positions may be occupied by the elephant.
            for idx,pos in preyers.items():
                new_field[pos]=0
                act=action[idx-1]     # let us define the action by a python dict
                pos_new=self.get_x_y(pos,act)
                if pos_new not in pos_set:
                    pos_set.add(pos_new)
                    new_preyers[idx]=pos_new
                elif pos not in pos_set:
                    pos_set.add(pos)
                    new_preyers[idx]=pos
                # else:
                #     status='crashed'
                #     print('ERROR! Preyer Crush!')
            for idx,pos in new_preyers.items():
                new_field[pos]=idx     # use the 'idx's to distinguish the preyers
            return new_field,ele_pos,new_preyers,status


    def apply_action(self,action):
        new_field,new_ele_pos,new_preyers,status =self._update_pos(action)
        # self.step_count+=1
        next_obs={
            'field':new_field,
            'ele_pos':new_ele_pos,
            'preyers':new_preyers,
        }
        reward=0    # temporary set reward to 0
        return FieldState(next_obs, self.next_player.other, action, status)

    def is_over(self):
        done=False
        info_list=[]
        winner=None
        if self._if_surrounded():
            done=True
            info_list.append('Surrounded!')
        if self._if_time_out():
            done=True
            info_list.append('Time out!')
        if self._if_ele_wins():
            done=True
            info_list.append('Ele Wins!')
        # if status=='crashed':
        #     done=True
        #     info_list.append('Crush!')
        if self._if_crashed():
            done=True
            info_list.append('Crush!')
        if done:
            winner=self.winner(info_list)
            return True,info_list,winner
        else:
            return False,[],winner

    def winner(self,info):
        winner=None
        if 'Surrounded!' in info:
            winner=Player.preyer
        elif 'Ele Wins!' in info or 'Crush!' in info:
            winner=Player.ele
        else:
            winner=None
        return winner

    def _if_crashed(self):
        pos_set={self.ele_pos,self.preyers[1],self.preyers[2],self.preyers[3]}
        if len(pos_set)<4:
            return True

    def _if_surrounded(self):
        surrounded_list=[]
        for _, pos in self.preyers.items():
            if distance.cityblock(self.ele_pos,pos)==1:
                surrounded_list.append(pos)
        N_preyers=3     # Note: the number of the preyers here is fixed
            
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
        field_size=len(self.field)
        if ele_pos in [
            (0,0),
            (field_size-1,field_size-1),
            (0,field_size-1),
            (field_size-1,0)
            ]:
            return True

    def _if_time_out(self):
        # if self.step_count>=15:
        #     return True
        # else:
        #     return False
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

        size=len(self.field)
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

    def synch(self):
        # 同化功能，可直接用赋值方法实现
        pass

    def legal_moves(self):
        if self.next_player==Player.ele:
            legal_moves=[]
            # for act in range(1,10):
            for act in [2,4,6,8]:
                pos_new=self.get_x_y(self.ele_pos,act)
                if pos_new!=self.ele_pos:
                    legal_moves.append(act)
            # legal_moves.append(5)
            return legal_moves

        elif self.next_player==Player.preyer:
            every_legal_moves={}
            for idx,preyer_pos in self.obs['preyers'].items():
                per_p_legal_moves=[]
                # for act in range(1,10):
                for act in [2,4,6,8]:
                    pos_new=self.get_x_y(preyer_pos,act)
                    if pos_new!=preyer_pos:
                        per_p_legal_moves.append(act)
                # per_p_legal_moves.append(5)
                every_legal_moves[idx]=per_p_legal_moves
            legal_moves=[]
            for move1 in every_legal_moves[1] :
                for move2 in every_legal_moves[2]:
                    for move3 in every_legal_moves[3]:
                        move=[move1,move2,move3]
                        legal_moves.append(move)
            return legal_moves