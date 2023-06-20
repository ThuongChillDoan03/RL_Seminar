import numpy as np
import random as rd
from numba import njit,jit

@njit()
def Calculate(arr_card_player):
   arr_point = np.where(arr_card_player == 1)[0]
   point_player = 0
   for idx in arr_point:
      if idx%13 >= 9:
         point_player += 10
      if idx%13 < 9:
         point_player += (idx%13 + 1)
   return point_player

@njit()
def getAgentSize():
   return 4

@njit()
def getActionSize():
   return 322

def visualizeEnv(env_state):
   dict_ = {}
   dict_["Bài úp trên bàn:"] = env_state[:len(np.where(env_state[0:52] != -1)[0])]
   dict_["Bài ngửa trên bàn:"] = env_state[52:(52+len(np.where(env_state[52:104] != -1)[0]))]
   dict_["Bài trên tay người thứ nhất:"] = np.where(env_state[109:161]==1)[0]
   dict_["Bài trên tay người thứ hai:"] = np.where(env_state[161:213]==1)[0]
   dict_["Bài trên tay người thứ ba:"] = np.where(env_state[213:265]==1)[0]
   dict_["Bài trên tay người thứ tư:"] = np.where(env_state[265:317]==1)[0]
   dict_["Người số mấy đang chơi:"] = env_state[104]
   dict_["Điểm ván nhỏ:"] = env_state[327:331]
   dict_["Điểm ván lớn:"] = env_state[317:321]
   dict_["Người còn lại trong bàn chơi:"] = env_state[321:325]
   dict_["Các bộ đã được đánh ra:"] = np.where(env_state[331:600]==1)[0]
   dict_["turn"] = env_state[325]

   return dict_

@njit()
def initEnv():
   env_state = np.full(603,0)
   env_state[:104] = -1
   temp = np.arange(52)
   np.random.shuffle(temp)
   env_state[109+temp[:7]] = 1
   env_state[161+temp[7:14]] = 1
   env_state[213+temp[14:21]] = 1
   env_state[265+temp[21:28]] = 1
   env_state[52] = temp[28]
   env_state[:23] = temp[29:] 

   env_state[105:109] = 1
   env_state[321:325] = 1

   for i_ in range(4):
      arr_card_player = env_state[(109+i_*52):(109+(i_+1)*52)]
      point_player = Calculate(arr_card_player)
      env_state[327+i_] = point_player
   env_state[600] = 1

   return env_state

@njit()
def book_run_hit(arr_check_connect):
   arr_card_connect = np.where(arr_check_connect == 1)[0]
   return arr_card_connect

@njit()
def getAgentState(env_state):
   P_state = np.full(127,0)
   Player_ID = env_state[104]
   P_state[:52] = env_state[(109+Player_ID*52):(109+(Player_ID+1)*52)]
   P_state[52:104] = env_state[52:104]
   for idx in range(4):
      if idx+Player_ID <= 3:
         P_state[104+idx] = env_state[321+idx+Player_ID]
         P_state[108+idx] = env_state[317+idx+Player_ID]
      if idx+Player_ID > 3:
         P_state[104+idx] = env_state[317+idx+Player_ID]
         P_state[108+idx] = env_state[313+idx+Player_ID]

   P_state[112] = env_state[326]
   arr_check_connect = env_state[331:600]
   arr_card_connect = book_run_hit(arr_check_connect)
   if len(arr_card_connect) > 0:
      P_state[113:(113+len(arr_card_connect))] = arr_card_connect
   P_state[125] = len(np.where(env_state[:52] != -1)[0])
   P_state[126] = env_state[602]

   return P_state.astype(np.float64)

@njit()
def straight_subsequences(arr):
   arr_return = []
   n = len(arr)
   for k in range(3,8):
      if n < k:
         arr_return = np.full((0,2),0)
         return arr_return
      
      for i in range(0, n-k+1):
         sub_arr = arr[i:i+k]
         if (np.max(sub_arr) - np.min(sub_arr) == k-1) and (int(sub_arr[0]//13) == int(sub_arr[k-1]//13)):
            arr_return.append([sub_arr[0], k])
   if len(arr_return) == 0:
      arr_return = np.full((0,2),0)
      return arr_return
   else:
      return np.array(arr_return)

@njit()
def breed_set(arr):
   arr_return_breed = []
   n = len(arr)
   for i_ in range(n):
      numbers_of_card_breed = 0
      points_left = 10
      for k_ in range(i_,n):
         if (arr[k_] - arr[i_]) % 13 == 0:
            numbers_of_card_breed += 1
            points_left -= (int(arr[k_]//13) + 1)
         if numbers_of_card_breed == 3 and (arr[k_] - arr[i_])%13==0:
            arr_return_breed.append([points_left - 1, arr[i_]%13])
         if numbers_of_card_breed == 4 and (arr[k_] - arr[i_])%13==0:
            arr_return_breed.append([-1, arr[i_]%13])
   if len(arr_return_breed) == 0:
      arr_return_breed = np.full((0,2),0)
      return arr_return_breed
   else:
      return np.array(arr_return_breed)


@njit()
def card_on_hand(arr_3):
   arr_card_on_hand = np.where(arr_3 == 1)[0]
   return arr_card_on_hand

@njit()
def check_join_sets(arr_card_on_board,arr_card_on_hand_1):
   arr_card_join = []
   for k_ in arr_card_on_hand_1:
      natural_parts = int(k_//13)
      for idx_ in arr_card_on_board:
         if idx_ < 44:
            natural_parts_run = int(idx_//11)
            if natural_parts_run == natural_parts:    #cùng chất
               if (k_%13)-(idx_%11)==3 or (idx_%11)-(k_%13)==1:   # nối dây
                  arr_card_join.append([k_,idx_])
         if 44 <= idx_ < 84:
            natural_parts_run = int((idx_-44)//10)
            if natural_parts_run == natural_parts:    
               if (k_%13)-((idx_-44)%10)==4 or ((idx_-44)%10)-(k_%13)==1: 
                  arr_card_join.append([k_,idx_])
         if 84 <= idx_ < 120:
            natural_parts_run = int((idx_-84)//9)
            if natural_parts_run == natural_parts: 
               if (k_%13)-((idx_-84)%9)==5 or ((idx_-84)%9)-(k_%13)==1:
                  arr_card_join.append([k_,idx_])
         if 120 <= idx_ < 152:
            natural_parts_run = int((idx_-120)//8)
            if natural_parts_run == natural_parts: 
               if (k_%13)-((idx_-120)%8)==6 or ((idx_-120)%8)-(k_%13)==1:
                  arr_card_join.append([k_,idx_])
         if 152 <= idx_ < 180:
            natural_parts_run = int((idx_-152)//7)
            if natural_parts_run == natural_parts:
               if (k_%13)-((idx_-152)%7)==7 or ((idx_-152)%7)-(k_%13)==1:
                  arr_card_join.append([k_,idx_])
         if 180 <= idx_ < 204:
            continue
         if 204 <= idx_ < 256:
            if (k_%13) == int((idx_-204)//4):
               arr_card_join.append([k_,idx_])
         if 256 <= idx_ < 269:
            continue
   if len(arr_card_join) == 0:
      arr_card_join = np.full((0,2),0)
      return arr_card_join
   else:
      return np.array(arr_card_join)

@njit()
def pick_card_upside_dow(arr_2):
   pick_upside_dow = np.where(arr_2 != -1)[0]
   arr_pick_upside_dow = arr_2[0:len(pick_upside_dow)]
   return arr_pick_upside_dow

@njit()
def getValidActions(P_state):
   if P_state[112] == 0:   #rút bài
      Valid_Actions_return = np.full(2,0)
      arr_2 = P_state[52:104]
      arr_pick_upside_dow = pick_card_upside_dow(arr_2)
      if len(arr_pick_upside_dow) > 0:
         Valid_Actions_return[1] = 1   #bộ ngửa
      if P_state[125] > 0:
         Valid_Actions_return[0] = 1
      return Valid_Actions_return.astype(np.int64)
   
   ###__________________________________check_chọn_phase___________________________###

   if P_state[112] == 1:   #Hỏi xem muốn đánh bài hay nối dây 
      Valid_Actions_return = np.full(2,1)
      arr_card_on_hand_1 = np.where(P_state[:52] == 1)[0]
      arr_place_run_book = np.where(P_state[113:123] != 0)[0]
      arr_card_on_board_2 = P_state[113:(113+len(arr_place_run_book))]
      arr_card_on_board = []
      for cards in arr_card_on_board_2:
         arr_card_on_board.append(int(cards))
      arr_card_on_board = np.array(arr_card_on_board)

      if len(arr_card_on_board) > 0:
         arr_card_join = check_join_sets(arr_card_on_board,arr_card_on_hand_1)
         if len(arr_card_join) == 0:
            Valid_Actions_return[1] = 0
         return Valid_Actions_return.astype(np.int64)
      if len(arr_card_on_board) == 0:
         Valid_Actions_return[1] = 0
         return Valid_Actions_return.astype(np.int64)
   ###_____________________________________________________________________________###
   if P_state[112] == 2:
      Valid_Actions_return = np.full(322,0)
      arr_3 = P_state[:52]
      arr_card_on_hand = card_on_hand(arr_3)
      Valid_Actions_return[0+arr_card_on_hand] = 1
      #______________________________action_run_card_______________________________#

      arr = np.where(P_state[:52] == 1)[0]
      arr_straight_subsequences = straight_subsequences(arr)
      if len(arr_straight_subsequences) > 0:
         for k_ in arr_straight_subsequences:
            natural_part = int(k_[0]//13)
            compensate = k_[0]%13
            if k_[1] == 3:
               Valid_Actions_return[52+natural_part*11+compensate] = 1
            if k_[1] == 4:
               Valid_Actions_return[96+natural_part*10+compensate] = 1
            if k_[1] == 5:
               Valid_Actions_return[136+natural_part*9+compensate] = 1
            if k_[1] == 6:
               Valid_Actions_return[172+natural_part*8+compensate] = 1
            if k_[1] == 7:
               Valid_Actions_return[204+natural_part*7+compensate] = 1
            if k_[1] == 8:
               Valid_Actions_return[232+natural_part*6+compensate] = 1
      #_____________________________action_book_card_____________________________#
      arr_breed = breed_set(arr)
      for i_ in arr_breed:
         if i_[0] == -1:
            Valid_Actions_return[308+i_[1]] = 1
            Valid_Actions_return[(256+i_[1]*4):(256+(i_[1]+1)*4)] = 1   
         if i_[0] != -1:
            Valid_Actions_return[256+i_[1]*4+i_[0]] = 1
      if P_state[126] == 1:
         Valid_Actions_return[321] = 1

      return Valid_Actions_return.astype(np.int64)
   if P_state[112] == 3:
      Valid_Actions_return = np.full(373,0)  #0:180 nối trên, 180:360:nối dưới
      arr_card_on_hand_1 = np.where(P_state[:52] == 1)[0]
      check_card_run = np.where(P_state[113:123] != 0)[0]
      arr_card_on_board_1 = P_state[113:(113+len(check_card_run))]
      arr_card_on_board = []
      for place in arr_card_on_board_1:
         arr_card_on_board.append(int(place))
      arr_card_on_board = np.array(arr_card_on_board)
      arr_card_join = check_join_sets(arr_card_on_board,arr_card_on_hand_1)
      for card_ in arr_card_join:
         if card_[1] < 180:
            if card_[1] < 44:
               run_can_choose = card_[1]%11
            if 44 <= card_[1] < 84:
               run_can_choose = (card_[1]-44)%10
            if 84 <= card_[1] < 120:
               run_can_choose = (card_[1]-84)%9
            if 120 <= card_[1] < 152:
               run_can_choose = (card_[1]-120)%8
            if 152 <= card_[1] < 180:
               run_can_choose = (card_[1]-152)%7
            if (card_[0]%13) > run_can_choose:
               Valid_Actions_return[card_[1]] = 1  #nối trên
            if (card_[0]%13) < run_can_choose:
               Valid_Actions_return[180+card_[1]] = 1 #nối dưới
         if 204 <= card_[1] < 256:
            choose_card = int((card_[1]-204)//4)
            Valid_Actions_return[360+choose_card] = 1

      return Valid_Actions_return.astype(np.int64)

@njit()
def check_status_player(arr_status):
   arr_player = np.where(arr_status == 1)[0]
   return arr_player

@njit()
def stepEnv(action,env_state):
   Player_ID = env_state[104]
   if env_state[600] == 1:
      check_winner_begin = np.where(env_state[327:331] == 50)[0]
      if len(check_winner_begin) >= 1:
         winner = check_winner_begin[0]
         env_state[105+winner] = 0
         env_state[327+winner] = 0
      env_state[600] = 0
   if np.sum(env_state[105:109]) == np.sum(env_state[321:325]):
      check_1 = 1
      check_2 = 1
      if env_state[326] == 0:
         if action == 0:
            arr_2 = env_state[:52]
            arr_pick_upside_dow = pick_card_upside_dow(arr_2)  #arr vị trí
            card_choose = arr_pick_upside_dow[-1]
            if len(arr_pick_upside_dow) >= 1:
               env_state[len(arr_pick_upside_dow)-1] = -1
            env_state[109+Player_ID*52+card_choose] = 1
            arr_card_player = env_state[(109+Player_ID*52):(109+(Player_ID+1)*52)]
            point_player = Calculate(arr_card_player)
            env_state[327+Player_ID] = point_player
            env_state[326] = 1
            check_1 = 0
         if action == 1:
            arr_2 = env_state[52:104]
            arr_pick_upside_dow = pick_card_upside_dow(arr_2)  #arr vị trí
            card_choose = arr_pick_upside_dow[-1]
            env_state[51+len(arr_pick_upside_dow)] = -1
            env_state[109+Player_ID*52+card_choose] = 1
            arr_card_player = env_state[(109+Player_ID*52):(109+(Player_ID+1)*52)]
            point_player = Calculate(arr_card_player)
            env_state[327+Player_ID] = point_player
            env_state[326] = 1
            check_1 = 0
      if env_state[326] == 1 and check_1 == 1:     #choose_phase
         if action == 0:
            env_state[326] = 2
            check_2 = 0
         if action == 1:
            env_state[326] = 3
            check_2 = 0
      if env_state[326] == 2 and check_2 == 1:  #phase_attack
         if action < 52:
            arr_check = np.where(env_state[52:104] != -1)[0]
            env_state[109+Player_ID*52+action] = 0
            env_state[52+len(arr_check)] = action
         if 52 <= action < 63:   #run_3
            choose = action - 52
            env_state[(109+Player_ID*52+choose):(112+Player_ID*52+choose)] = 0
         if 63 <= action < 74:
            choose = action - 63
            env_state[(122+Player_ID*52+choose):(125+Player_ID*52+choose)] = 0
         if 74 <= action < 85:
            choose = action - 74
            env_state[(135+Player_ID*52+choose):(138+Player_ID*52+choose)] = 0
         if 85 <= action < 96:
            choose = action - 85
            env_state[(148+Player_ID*52+choose):(151+Player_ID*52+choose)] = 0
         if 96 <= action < 106:  #run_4
            choose = action - 96
            env_state[(109+Player_ID*52+choose):(113+Player_ID*52+choose)] = 0
         if 106 <= action < 116: 
            choose = action - 106
            env_state[(122+Player_ID*52+choose):(116+Player_ID*52+choose)] = 0
         if 116 <= action < 126:
            choose = action - 116
            env_state[(135+Player_ID*52+choose):(139+Player_ID*52+choose)] = 0
         if 126 <= action < 136:
            choose = action - 126
            env_state[(148+Player_ID*52+choose):(152+Player_ID*52+choose)] = 0
         if 136 <= action < 145: #run_5
            choose = action - 136
            env_state[(109+Player_ID*52+choose):(114+Player_ID*52+choose)] = 0
         if 145 <= action < 154:
            choose = action - 145
            env_state[(122+Player_ID*52+choose):(127+Player_ID*52+choose)] = 0
         if 154 <= action < 163:
            choose = action - 154
            env_state[(135+Player_ID*52+choose):(140+Player_ID*52+choose)] = 0
         if 163 <= action < 172:
            choose = action - 163
            env_state[(148+Player_ID*52+choose):(153+Player_ID*52+choose)] = 0
         if 172 <= action < 180: #run_6
            choose = action - 172
            env_state[(109+Player_ID*52+choose):(115+Player_ID*52+choose)] = 0 #run_6
         if 180 <= action < 188:
            choose = action - 180
            env_state[(122+Player_ID*52+choose):(128+Player_ID*52+choose)] = 0
         if 188 <= action < 196:
            choose = action - 188
            env_state[(135+Player_ID*52+choose):(141+Player_ID*52+choose)] = 0
         if 196 <= action < 204:
            choose = action - 196
            env_state[(148+Player_ID*52+choose):(154+Player_ID*52+choose)] = 0
         if 204 <= action < 211: #run_7
            choose = action - 204
            env_state[(109+Player_ID*52+choose):(116+Player_ID*52+choose)] = 0
         if 211 <= action < 218:
            choose = action - 211
            env_state[(122+Player_ID*52+choose):(129+Player_ID*52+choose)] = 0 
         if 218 <= action < 225:
            choose = action - 218
            env_state[(135+Player_ID*52+choose):(142+Player_ID*52+choose)] = 0
         if 225 <= action < 232:
            choose = action - 225
            env_state[(148+Player_ID*52+choose):(155+Player_ID*52+choose)] = 0
         if 232 <= action < 238: #run_8
            choose = action - 232
            env_state[(109+Player_ID*52+choose):(117+Player_ID*52+choose)] = 0
         if 238 <= action < 244:
            choose = action - 238
            env_state[(122+Player_ID*52+choose):(130+Player_ID*52+choose)] = 0 
         if 244 <= action < 250:
            choose = action - 244
            env_state[(135+Player_ID*52+choose):(143+Player_ID*52+choose)] = 0
         if 250 <= action < 256:
            choose = action - 250
            env_state[(148+Player_ID*52+choose):(156+Player_ID*52+choose)] = 0
         if 256 <= action < 308:       #book_3
            ix_choose = action - 256
            natural_choose = int(ix_choose//4)
            part_choose = ix_choose%4
            for i_ in range(4):
               if i_ != part_choose:
                  env_state[109+Player_ID*52+i_*13+natural_choose] = 0
         if 308 <= action < 321:       #book_4
            choose = action - 308
            for i_ in range(4):
               env_state[109+Player_ID*52+i_*13+choose] = 0
         if action == 321:
            env_state[602] += 0
         if 52 <= action < 321 :
            env_state[279+action] = 1
         arr_card_player = env_state[(109+Player_ID*52):(109+(Player_ID+1)*52)]
         point_player = Calculate(arr_card_player)
         env_state[327+Player_ID] = point_player
         env_state[326] = 0
         env_state[325] += 1
      if env_state[326] == 3 and check_2 == 1:  #nối dây(HIT)
         if action < 44:
            color_card = int(action//11)
            place_card = action%11
            env_state[331+action] = 0
            env_state[375+10*color_card+place_card] = 1
            env_state[112+Player_ID*52+13*color_card+place_card] = 0 
         if 44 <= action < 84:
            color_card = int((action-44)//10)
            place_card = (action-44)%10
            env_state[331+action] = 0
            env_state[415+9*color_card+place_card] = 1
            env_state[113+Player_ID*52+13*color_card+place_card] = 0 
         if 84 <= action < 120:
            color_card = int((action-84)//9)
            place_card = (action-84)%9
            env_state[331+action] = 0
            env_state[451+8*color_card+place_card] = 1
            env_state[114+Player_ID*52+13*color_card+place_card] = 0 
         if 120 <= action < 152:
            color_card = int((action-120)//8)
            place_card = (action-120)%8
            env_state[331+action] = 0
            env_state[483+7*color_card+place_card] = 1
            env_state[115+Player_ID*52+13*color_card+place_card] = 0 
         if 152 <= action < 180:
            color_card = int((action-152)//7)
            place_card = (action-152)%7
            env_state[331+action] = 0
            env_state[511+6*color_card+place_card] = 1
            env_state[116+Player_ID*52+13*color_card+place_card] = 0 
         if 180 <= action < 224:
            color_card = int((action-180)//11)
            place_card = (action-180)%11
            env_state[151+action] = 0
            env_state[374+10*color_card+place_card] = 1
            env_state[108+Player_ID*52+13*color_card+place_card] = 0
         if 224 <= action < 264:
            color_card = int((action-224)//10)
            place_card = (action-224)%10
            env_state[151+action] = 0
            env_state[414+9*color_card+place_card] = 1
            env_state[108+Player_ID*52+13*color_card+place_card] = 0
         if 264 <= action < 300:
            color_card = int((action-264)//9)
            place_card = (action-264)%9
            env_state[151+action] = 0
            env_state[450+8*color_card+place_card] = 1
            env_state[108+Player_ID*52+13*color_card+place_card] = 0
         if 300 <= action < 332:
            color_card = int((action-300)//8)
            place_card = (action-300)%8
            env_state[151+action] = 0
            env_state[482+7*color_card+place_card] = 1
            env_state[108+Player_ID*52+13*color_card+place_card] = 0
         if 332 <= action < 360:
            color_card = int((action-332)//7)
            place_card = (action-332)%7
            env_state[151+action] = 0
            env_state[510+6*color_card+place_card] = 1
            env_state[108+Player_ID*52+13*color_card+place_card] = 0
         if 360 <= action < 373:
            env_state[227+action] = 1
            env_state[(535+(action-360)*4):(535+(action-359)*4)] = 0
            for jx in range(4):
               env_state[109+Player_ID*52+jx*13+(action-360)] = 0

         arr_card_player = env_state[(109+Player_ID*52):(109+(Player_ID+1)*52)]
         point_player = Calculate(arr_card_player)
         env_state[327+Player_ID] = point_player
         env_state[326] = 2

   ###__________________________________________________CHECK_CARD_ON_HAND_PLAYER___________________________________________________###
   arr_card_player = env_state[(109+Player_ID*52):(109+(Player_ID+1)*52)]
   check_len_card_on_hand = np.where(arr_card_player == 1)[0]
   if len(check_len_card_on_hand) == 0:
      env_state[105+Player_ID] = 0
   ###__________________________________________________CHECK_CARD_UP_END_ROUND__________________________________________________________###
   check_arr_card = np.where(env_state[:52] != -1)[0]
   player_on_board = np.where(env_state[321:325]==1)[0]
   if (len(check_arr_card) == 0) or (env_state[325]-env_state[601] == len(player_on_board)*10):
      arr_status_player = np.where(env_state[105:109] == 1)[0]
      check_win_arr = env_state[327+arr_status_player]
      if len(check_win_arr) > 0:
         min_point = np.min(check_win_arr)
         arr_winner = np.where(check_win_arr == min_point)[0]
         winner = arr_winner[0]
         env_state[105+winner] = 0
         env_state[327+winner] = 0
      env_state[601] = env_state[325]
   ###__________________________________________________RESET_ROUND_________________________________________________________________###
   check_3 = 1
   if np.sum(env_state[105:109]) < np.sum(env_state[321:325]):   
      arr_status = env_state[105:109]
      arr_player = check_status_player(arr_status)
      for player_ in arr_player:
         env_state[317+player_] += env_state[327+player_]
         if env_state[317+player_] >= 100:
            env_state[321+player_] = 0 # loại ra khỏi bàn chơi
            env_state[327+player_] = 0 # xoá điểm ván nhỏ
      arr_players_are_still_on_the_board = env_state[321:325]
      players_are_still_on_the_board = np.where(arr_players_are_still_on_the_board == 1)[0]
      if len(players_are_still_on_the_board) >= 2: #RESET ROUND
         env_state[0:104] = -1
         env_state[109:317] = 0
         env_state[331:600] = 0
         env_state[600] = 1
         temp = np.arange(52)
         np.random.shuffle(temp) 
         for playerx in range(len(players_are_still_on_the_board)):
            env_state[109+players_are_still_on_the_board[playerx]*52+temp[(playerx*7):((playerx+1)*7)]] = 1
            arr_card_player = env_state[(109+players_are_still_on_the_board[playerx]*52):(109+(players_are_still_on_the_board[playerx]+1)*52)]
            point_player = Calculate(arr_card_player)
            env_state[327+players_are_still_on_the_board[playerx]] = point_player
         env_state[52] = temp[len(players_are_still_on_the_board)*7]
         env_state[0:(51-len(players_are_still_on_the_board)*7)] = temp[(len(players_are_still_on_the_board)*7+1):]
         env_state[105:109] = env_state[321:325]
         env_state[326] = 0
         check_3 = 0
         env_state[104] = players_are_still_on_the_board[0]
      else:
         env_state[602] = 1

   ###_________________________________________Check_Người_Nhận_Action_Tiếp_Theo_____________________________________________###
   if check_3 == 1:
      arr_the_following_status = env_state[105:109]
      the_following_status = np.where(arr_the_following_status == 1)[0]
      if len(the_following_status) >= 2:
         if env_state[326] != 0:
            env_state[104] += 0
         if env_state[326] == 0:
            next_player = np.where(the_following_status > env_state[104])[0]
            if len(next_player) == 0:
               env_state[104] = the_following_status[0]
            if len(next_player) > 0:
               place = next_player[0]
               env_state[104] = the_following_status[place]

   return env_state

@njit()
def checkEnded(env_state):
   arr_status = env_state[321:325]
   check_win_player = np.where(arr_status == 1)[0]
   if env_state[602] == 1 and len(check_win_player) >= 2:
      return -1
   
   if len(check_win_player) == 1:
      return check_win_player[0]
   else:
      return -1

@njit()
def getReward(P_state):
   if P_state[126] == 0:
      return -1
   else:
      arr_individual_score_win = P_state[104:108]
      individual_score_win = np.where(arr_individual_score_win == 1)[0]
      if len(individual_score_win) >= 2:
         return 0
      if len(individual_score_win) == 1:
         if individual_score_win[0] == 0:
            return 1
         if individual_score_win[0] != 0:
            return 0

@njit()
def getStateSize():
   return 127

def randomBot(P_state, perData):
   validActions = getValidActions(P_state)
   validActions = np.where(validActions==1)[0]
   idx = np.random.randint(0, len(validActions))
   return validActions[idx], perData

def one_game(listAgent, perData):#----------------------------------------------
   env_state = initEnv()
   winner = -1
   while env_state[325] < 5000:
      pIdx = env_state[104]
      # if turn != pIdx :
      #   turn = pIdx
      #   print("--------Turn =", env_[113])
      action, perData = listAgent[pIdx](getAgentState(env_state), perData)
      # print(action)
      stepEnv(action, env_state)
      # print(visualizeEnv(env_state))
      winner = checkEnded(env_state)
      if winner != -1:
         break
   
   env_state[602] = 1
   for pIdx in range(4):
      env_state[104] = pIdx
      action, perData = listAgent[pIdx](getAgentState(env_state), perData)
   return winner, perData

def normal_main(listAgent, times, perData):#------------------------------------
   if len(listAgent) != 4:
      raise Exception('Hệ thống chỉ cho phép có đúng 4 người chơi!!!')
   numWin = [0, 0, 0, 0, 0]
   pIdOrder = np.arange(4)
   for _ in range(times):
      # if printMode and _ != 0 and _ % k == 0:
      #     print(_, numWin)
      np.random.shuffle(pIdOrder)
      # print(pIdOrder)
      shuffledListAgent = [listAgent[i] for i in pIdOrder]
      winner, perData = one_game(shuffledListAgent, perData)
      if winner == -1:
         numWin[4] += 1
      else:
         numWin[pIdOrder[winner]] += 1
  
   print(_+1, numWin)

   return numWin, perData

@njit()
def numbaRandomBot(P_state, perData):
   validActions = getValidActions(P_state)
   validActions = np.where(validActions==1)[0]
   idx = np.random.randint(0, len(validActions))
   return validActions[idx], perData

@njit()
def numba_one_game(p0, p1, p2, p3, perData, pIdOrder):
   env_state = initEnv()

   winner = -1
   while env_state[325] < 5000:
      pIdx = env_state[104]
      if pIdOrder[pIdx] == 0:
         action, perData = p0(getAgentState(env_state), perData)
      elif pIdOrder[pIdx] == 1:
         action, perData = p1(getAgentState(env_state), perData)
      elif pIdOrder[pIdx] == 2:
         action, perData = p2(getAgentState(env_state), perData)
      elif pIdOrder[pIdx] == 3:
         action, perData = p3(getAgentState(env_state), perData)
      
      stepEnv(action, env_state)
      winner = checkEnded(env_state)
      if winner != -1:
         break
   
   env_state[602] = 1
   for pIdx in range(4):
      env_state[104] = pIdx
      if pIdOrder[pIdx] == 0:
         action, perData = p0(getAgentState(env_state), perData)
      elif pIdOrder[pIdx] == 1:
         action, perData = p1(getAgentState(env_state), perData)
      elif pIdOrder[pIdx] == 2:
         action, perData = p2(getAgentState(env_state), perData)
      elif pIdOrder[pIdx] == 3:
         action, perData = p3(getAgentState(env_state), perData)
   return winner, perData

@njit()
def numba_main(p0, p1, p2, p3, times, perData):
   numWin = np.full(5, 0)
   pIdOrder = np.arange(4)
   for _ in range(times):
      # if printMode and _ != 0 and _ % k == 0:
      #   print(_, numWin)
      np.random.shuffle(pIdOrder)
      winner, perData = numba_one_game(p0, p1, p2, p3, perData, pIdOrder)

      if winner == -1:
         numWin[4] += 1
      else:
         numWin[pIdOrder[int(winner)]] += 1
   return numWin, perData

@jit()
def one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
   env_state = initEnv()
   while env_state[325] < 5000:
      idx = env_state[104]
      player_state = getAgentState(env_state)
      list_action = getValidActions(player_state)

      if list_other[idx] == -1:
         action, per_player = p0(player_state,per_player)
      elif list_other[idx] == 1:
         action, per1 = p1(player_state,per1)
      elif list_other[idx] == 2:
         action, per2 = p2(player_state,per2)
      elif list_other[idx] == 3:
         action, per3 = p3(player_state,per3)

      if list_action[action] != 1:
         raise Exception('Action không hợp lệ')
      stepEnv(action, env_state)
      if checkEnded(env_state) != -1:
         break

   turn = env_state[325]
   env_state[602] = 1
   for idx in range(4):
      env_state[104] = idx
      if list_other[idx] == -1:
         p_state = getAgentState(env_state)
         act, per_player = p0(p_state, per_player)

   env_state[325] = turn
   winner = 0
   if np.where(list_other == -1)[0] == checkEnded(env_state): winner = 1
   else: winner = 0
   return winner, per_player

@njit()
def random_Env(P_state, per):
   arr_action = getValidActions(P_state)
   arr_action = np.where(arr_action == 1)[0]
   act_idx = np.random.randint(0, len(arr_action))
   return arr_action[act_idx], per

@jit()
def n_game_numba(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
   win = 0
   for _n in range(num_game):
      np.random.shuffle(list_other)
      winner,per_player = one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3)
      win += winner
   return win, per_player

# import importlib.util, json, sys
# from setup import SHOT_PATH

# def load_module_player(player):
#    return importlib.util.spec_from_file_location('Agent_player', f"{SHOT_PATH}Agent/{player}/Agent_player.py").loader.load_module()

# def numba_main_2(p0, n_game, per_player, level, *args):
#    list_other = np.array([1, 2, 3, -1])
#    if level == 0:
#       per_agent_env = np.array([0])
#       return n_game_numba(p0, n_game, per_player, list_other, per_agent_env, per_agent_env, per_agent_env, random_Env, random_Env, random_Env)
#    else:
#       env_name = sys.argv[1]
#       if len(args) > 0:
#          dict_level = json.load(open(f'{SHOT_PATH}Log/check_system_about_level.json'))
#       else:
#          dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))

#       if str(level) not in dict_level[env_name]:
#          raise Exception('Hiện tại không có level này') 

#       lst_agent_level = dict_level[env_name][str(level)][2]
#       p1 = load_module_player(lst_agent_level[0]).Test
#       p2 = load_module_player(lst_agent_level[1]).Test
#       p3 = load_module_player(lst_agent_level[2]).Test
#       per_level = []
#       for id in range(getAgentSize()-1):
#          data_agent_env = list(np.load(f'{SHOT_PATH}Agent/{lst_agent_level[id]}/Data/{env_name}_{level}/Train.npy',allow_pickle=True))
#          per_level.append(data_agent_env)

#       return n_game_numba(p0, n_game, per_player, list_other, per_level[0], per_level[1], per_level[2], p1, p2, p3)

perData = [np.zeros(3),np.zeros(getStateSize()),np.zeros(getStateSize()),np.zeros(getStateSize())]
##_List_per_data_agent_test

@njit()
def agent_find_value(P_state,perData):
   actions = getValidActions(P_state)
   output = np.random.rand(getActionSize()) * actions + actions
   action = np.argmax(output)
   win = getReward(P_state)
   if perData[0][0] < 1000:
      if win == 1:
         perData[1] = np.minimum(perData[1],P_state)
         perData[2] = np.maximum(perData[2],P_state)
         perData[3] = P_state
         perData[0][0] += 1  #số lần win agent
   else:
      value = np.zeros(getStateSize())
      for id in range(getStateSize()):
         sample = np.zeros(getStateSize()) + perData[3]
         ifmin, ifmax = 0,0
         sample[id] = perData[1][id]         #Minimum
         if getReward(sample) == 1:
               ifmin = 1
         sample[id] = perData[2][id]
         if getReward(sample) == 1:
               ifmax = 1
         if ifmax > ifmin:
               value[id] = 1
         if ifmin > ifmax:
               value[id] = -1
      perData[1] = value
   perData[0][1] += 1 
   if win != -1:
      if perData[0][2] == 0:      
         perData[0][2] = perData[0][1]
      else:
         perData[0][2] = np.maximum(perData[0][1],perData[0][2])
      perData[0][1] = 0
   return action, perData

listAgent = [randomBot,agent_find_value,randomBot,randomBot]

normal_main(listAgent, 100, perData)
print(numba_main(numbaRandomBot,agent_find_value,numbaRandomBot,numbaRandomBot,100,perData))