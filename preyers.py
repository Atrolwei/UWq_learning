import enum
from agents_base import Agent
from field_state import FieldState
import random
import copy


class GameResult(enum.Enum):
    loss = 1
    draw = 2
    win = 3


class Preyer(Agent):
    def __init__(self,name,idx,obs):
        super(Preyer,self).__init__()
        self.name=name
        self.index=idx
        last_action=[5,5,5]
        self.last_action=last_action
        
        self.field_state=FieldState(obs,Player.preyer,last_action)


    def synch(self,obs,action,syn_mode='tacit'):
        if syn_mode=='explicit':
            self.field_state=FieldState(obs,Player.preyer,action)
        elif syn_mode=='tacit':
            # 默契同化
            new_field_state=self.field_state.apply_action(self.last_action)
            self.field_state=new_field_state.apply_action(action)
            # debug here
            if self.field_state.obs['field'].all()==obs['field'].all():
                print('synch OK!')

            else:
                print('synch error!')


    def sample(self,obs):
        winning_moves=[]
        draw_moves=[]
        losing_moves=[]
        for possible_move in self.field_state.legal_moves():     # 此处相当于 env直接记录了该轮到谁了，所以直接调用就可以
                                                                 # 另外，可以编写一个函数，直接给出preyers的动作组合
            next_field_state=self.field_state.apply_action(possible_move)
            depth=0
            opponent_best_outcome=self.best_result(next_field_state,depth)     # 对手的最佳应对结果

            our_best_outcome=self.reversed_game_result(opponent_best_outcome) # 对手最佳应对结果的反面就是我方行动的最大收获

            if our_best_outcome=="win":
                winning_moves.append(possible_move)
            elif our_best_outcome=="draw":
                draw_moves.append(possible_move)
            else:
                losing_moves.append(possible_move)
        move_choice=None
        if winning_moves:
            move_choice=random.choice(winning_moves)     # 这里按照胜利随机选取行动是有危险的
                                                    # 解决方法：首先对行动赋予分值，排序选取。分数相同时，出现不确定性
                                                    # 出现不确定性，评估后续动作是否能有同化解，如有则无需通信，继续动作
                                                    # 若无同化解，则选择能接近设定邻居的解
        elif draw_moves:
            move_choice=random.choice(draw_moves)
        elif losing_moves:
            move_choice=random.choice(losing_moves)
        self.last_action=move_choice    # 记录上一拍的动作，用于默契推算
        return move_choice


    def reversed_game_result(self,game_result):
        if game_result == GameResult.loss:
            return game_result.win
        if game_result == GameResult.win:
            return game_result.loss
        return GameResult.draw


    def best_result(self,field_state,depth):
        done,info,winner=field_state.is_over()           # 如果游戏结束，则可以根据info评估谁赢了
        if done or depth>=6:
            if winner==field_state.next_player:     # winner==黑
                return GameResult.win
            elif winner is None:
                # A draw.
                return GameResult.draw
            else:
                # Opponent won.
                return GameResult.loss      

        # 如果尚未终局，需要向前搜索状态
        depth+=1
        '''
        下方代码对我方任一可选动作，均向前推理一步
        '''
        best_result_so_far=GameResult.loss
        for candidate_action in field_state.legal_moves():    # 己方的可选合法动作
            next_field_state=field_state.apply_action(candidate_action)     # 己方落子，nextplay为对方（白）
            # print(f'Player: {field_state.next_player},\nField:{next_field_state.field}')
            opponent_best_result=self.best_result(next_field_state,depth)     # 发现棋局已经结束，
            our_result=self.reversed_game_result(opponent_best_result)
            if our_result.value>best_result_so_far.value:       # 由于之前给result搞成了 枚举 Enum，因而可以直接比较
                best_result_so_far=our_result
        return best_result_so_far


        def predict(self,obs):
            action=self.f(obs)
            return action

        def f(self,obs):
            return 5
            
        def learn(self):
            pass



    
    