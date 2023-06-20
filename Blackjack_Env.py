import numpy as np
import math as m
from numba import njit, jit_module, generated_jit, jit
import random as rd
import time

@njit()
def initEnv():
    env_state = np.full(202,0)
    # index[201]: lá thứ 2 cái úp xuống.
    # index[200]: agent nào đang đc nhận action
    env_state[:9] = 32
    env_state[9] = 128
    env_state[10] = 100000
    env_state[11:17] = 1000
    for idx in range(7):
        env_state[23+idx*2] = 1
    
    return env_state

@njit()
def Caculus_card(arr_card_on_hand):
    point = 0
    if np.sum(arr_card_on_hand) == 2 and arr_card_on_hand[0] == 1 and arr_card_on_hand[9] == 1:
        point = 21
    else:
        arr_unchanged_point = arr_card_on_hand[1:]
        point_1 = 0
        for idx in range(len(arr_unchanged_point)):
            point_1 += arr_unchanged_point[idx]*(idx+2)
        if point_1 > 10:
            point = point_1 + arr_card_on_hand[0]
        if point_1 <= 10:
            point = point_1 + arr_card_on_hand[0]*11
    return point

@njit()
def getAgentState(env_state):
    p_id = int(env_state[199]//2)   
    p_state = np.full(175,0)
    p_state[:10] = env_state[:10]
    # tại vị trí mà agent đang nhận được nhận action(giá trị hiển thị từ p_state luôn coi là vị trí thứ a0)
    for idx in range(7):
        if (p_id+idx) <= 6:
            p_state[(10+20*idx):(10+20*(idx+1))] = env_state[(37+(p_id+idx)*20):(37+(p_id+idx+1)*20)]
            p_state[(150+2*idx):(150+2*(idx+1))] = env_state[(177+(p_id+idx)*2):(177+(p_id+idx+1)*2)]
            p_state[164+idx] = env_state[10+p_id+idx]
        elif (p_id+idx) > 6:
            p_state[(10+20*idx):(10+20*(idx+1))] = env_state[(37+(p_id+idx-7)*20):(37+(p_id+idx-6)*20)]
            p_state[(150+2*idx):(150+2*(idx+1))] = env_state[(177+(p_id+idx-7)*2):(177+(p_id+idx-6)*2)]
            p_state[164+idx] = env_state[3+p_id+idx]
    p_state[171] = env_state[197]
    p_state[172] = env_state[199]
    if p_id == 0:
        p_state[173] = 1
    else:
        p_state[173] = env_state[16+p_id]
    p_state[174] = env_state[198]

    return p_state.astype(np.float64)

@njit()
def getActionSize():
    return 9

@njit()
def getValidActions(p_state):
    Valid_Actions_return = np.full(9,0)
    set_card_one = p_state[10:20]
    set_card_two = p_state[20:30]
    check_bet = p_state[173]
    if check_bet == 0:
        if p_state[164] >= 100:
            Valid_Actions_return[0:4] = 1
        elif 50 <= p_state[164] < 100:
            Valid_Actions_return[0:3] = 1
        elif 20 <= p_state[164] < 50:
            Valid_Actions_return[0:2] = 1
        elif 10 <= p_state[164] < 20:
            Valid_Actions_return[0] = 1
        elif p_state < 10:
            Valid_Actions_return[4] = 1
    if check_bet == 1:
        Valid_Actions_return[8] = 1
    if check_bet != 0 and check_bet != 1:
        Valid_Actions_return[4] = 1
        if np.sum(set_card_one) == 2 and np.sum(set_card_two) == 0 and len(np.where(set_card_one==2)[0]) == 1:
            if p_state[164] >= p_state[173]:
                Valid_Actions_return[4:8] = 1
        if np.sum(set_card_one) == 2 and np.sum(set_card_two) == 0:
            if p_state[164] >= p_state[173]:
                Valid_Actions_return[4:7] = 1
        if np.sum(set_card_one) + np.sum(set_card_two) > 2:
            Valid_Actions_return[4:6] = 1
        if np.sum(set_card_one) + np.sum(set_card_two) == 0:
            Valid_Actions_return[4] = 1
        
    return Valid_Actions_return.astype(np.int64)

# Hàm random xác suất theo trọng số
@njit()
def weighted_random(env_state):
    card_weighted_random = env_state[:10]
    # rate_random = card_weighted_random/np.sum(card_weighted_random)
    can_choose = np.where(card_weighted_random > 0)[0]
    choice_card_place = np.random.choice(can_choose)
    env_state[choice_card_place] -= 1
    return choice_card_place


##______Hàm trả ra list các người chơi có blackjack_____##
@njit()
def player_blackjack(env_state):
    arr_point = env_state[177:191]
    arr_blackjack = np.full(14,0)
    for ind in range(14):
        if np.sum(env_state[(37+10*ind):(37+10*(ind+1))]) == 2 and arr_point[ind] == 21:
            arr_blackjack[ind] = 1
    list_id_blackjack = np.where(arr_blackjack == 1)[0]
    return list_id_blackjack

###_____Hàm trả ra số tiền được mất của người chơi____###
@njit()
def coin_return(list_id_blackjack, env_state):
    list_coin_return = np.full(7,0)
    list_point = env_state[177:191]
    out_point = np.where(list_point > 21)[0]
    number_of_set_id = np.full(7,0)
    for i in range(7):
        point_set_one = list_point[2*i]
        point_set_two = list_point[1+2*i]
        if point_set_one > 0:
            number_of_set_id[i] += 1
        if point_set_two > 0:
            number_of_set_id[i] += 1 

    if len(out_point) > 0:  # có người bị out điểm
        if len(list_id_blackjack) > 0:
            if out_point[0] == 0:   #nhà cái bị out điểm
                for ind in range(2,14):
                    if 0 < list_point[ind] <= 21:
                        list_coin_return[int(ind//2)] += env_state[16+int(ind//2)]//number_of_set_id[int(ind)//2]
                for ix in range(len(list_id_blackjack)):
                    list_coin_return[int(list_id_blackjack[ix]//2)] *= 1.5
                bot_coin_cal = np.sum(list_coin_return[1:])
                list_coin_return[0] = -bot_coin_cal
            if out_point[0] != 0 and list_id_blackjack[0] != 0: #cái có điểm ko out cũng ko có blj
                for ind in range(2,14):
                    if list_point[0] < list_point[ind] <= 21:
                        list_coin_return[int(ind//2)] += env_state[16+int(ind//2)]//number_of_set_id[int(ind)//2]
                    if 0 < list_point[ind] < list_point[0]:
                        list_coin_return[int(ind//2)] -= env_state[16+int(ind//2)]//number_of_set_id[int(ind)//2]
                for indx in range(len(list_id_blackjack)):
                    list_coin_return[int(list_id_blackjack[indx]//2)] *= 1.5
                for check in range(len(out_point)):
                    list_coin_return[int(out_point[check]//2)] -= env_state[16+int(out_point[check]//2)]//number_of_set_id[int(out_point[check]//2)]
                bot_coin_cal = np.sum(list_coin_return[1:])
                list_coin_return[0] = -bot_coin_cal
            if list_id_blackjack[0] == 0:   #cái có blj
                arr_id_player_blj = list_id_blackjack//2
                list_pay = np.full(7,1)
                for ix in range(7):
                    if ix in arr_id_player_blj:
                        list_pay[ix] = 0
                list_pay_choose = np.where(list_pay == 1)[0]    #list ko có blj
                for pay in range(len(list_pay_choose)):
                    list_coin_return[list_pay_choose[pay]] -= 1.5*(env_state[16+list_pay_choose[pay]])
                bot_coin_cal = np.sum(list_coin_return[1:])
                list_coin_return[0] = -bot_coin_cal
        if len(list_id_blackjack) == 0: #không có ai có blackjack
            if out_point[0]  == 0:  #cái bị out điểm
                for ind in range(2,14):
                    if 0 < list_point[ind] <= 21:
                        list_coin_return[int(ind//2)] += env_state[16+int(ind//2)]//number_of_set_id[int(ind)//2] 
                bot_coin_cal = np.sum(list_coin_return[1:])
                list_coin_return[0] = -bot_coin_cal
            if out_point[0] != 0:   #cái không bị out điểm
                for ind in range(2,14):
                    if list_point[0] < list_point[ind] <= 21:
                        list_coin_return[int(ind//2)] += env_state[16+int(ind//2)]//number_of_set_id[int(ind)//2]
                    if 0 < list_point[ind] < list_point[0]:
                        list_coin_return[int(ind//2)] -= env_state[16+int(ind//2)]//number_of_set_id[int(ind)//2]
                for check in range(len(out_point)):
                    list_coin_return[int(out_point[check]//2)] -= env_state[16+int(out_point[check]//2)]//number_of_set_id[int(out_point[check]//2)]
                bot_coin_cal = np.sum(list_coin_return[1:])
                list_coin_return[0] = -bot_coin_cal
    if len(out_point) == 0:
        if len(list_id_blackjack) > 0:  #có blj
            if list_id_blackjack[0] == 0:
                arr_id_player_blj = list_id_blackjack//2
                list_pay = np.full(7,1)
                for ix in range(7):
                    if ix in arr_id_player_blj:
                        list_pay[ix] = 0
                list_pay_choose = np.where(list_pay == 1)[0]    #list ko có blj
                for pay in range(len(list_pay_choose)):
                    list_coin_return[list_pay_choose[pay]] -= 1.5*(env_state[16+list_pay_choose[pay]])
                bot_coin_cal = np.sum(list_coin_return[1:])
                list_coin_return[0] = -bot_coin_cal
            if list_id_blackjack[0] != 0:
                arr_id_player_blj = list_id_blackjack//2
                for ind in range(2,14):
                    if list_point[0] < list_point[ind] <= 21:
                        list_coin_return[int(ind//2)] += (env_state[16+int(ind//2)]//number_of_set_id[int(ind)//2])
                    if 0 < list_point[ind] < list_point[0]:
                        list_coin_return[int(ind//2)] -= (env_state[16+int(ind//2)]//number_of_set_id[int(ind)//2])
                for idx in range(len(arr_id_player_blj)):
                    list_coin_return[arr_id_player_blj[idx]] *= 1.5
                bot_coin_cal = np.sum(list_coin_return[1:])
                list_coin_return[0] = -bot_coin_cal
        if len(list_id_blackjack) == 0: #ko có blj
            for index in range(2,14):
                if list_point[0] < list_point[index] <= 21:
                    list_coin_return[int(index//2)] += env_state[16+int(index//2)]//number_of_set_id[int(index)//2]
                if 0 < list_point[index] < list_point[0]:
                    list_coin_return[int(index//2)] -= env_state[16+int(index//2)]//number_of_set_id[int(index)//2]
            bot_coin_cal = np.sum(list_coin_return[1:])
            list_coin_return[0] = -bot_coin_cal

    return list_coin_return

# Hàm chuyển trạng thái player
@njit()
def next_id_player(env_state):
    arr_status_player = env_state[23:37]
    id_player_cur = env_state[199]
    arr_status_player_place = np.where(arr_status_player==1)[0]
    after_id_player = np.where(arr_status_player_place > id_player_cur)[0]
    if len(after_id_player) == 0:
        next_id = arr_status_player_place[0]
    if len(after_id_player) > 0:
        place = after_id_player[0]
        next_id = arr_status_player_place[place]

    return next_id

@njit()
def stepEnv(action,env_state):
    player_id = env_state[199]
    # Card_on_hand = env_state[(37+10*player_id):(37+10*(player_id+1))]
    # Point = env_state[171+player_id]
    remaining = np.sum(env_state[:10])
    if remaining <= 8:
        env_state[:10] = [32,32,32,32,32,32,32,32,32,128]
    level_bet = np.full(4,0)    #Arr_mức_cược
    level_bet[0] = 10
    level_bet[1] = 20
    level_bet[2] = 50
    level_bet[3] = 100

    if player_id >= 2:
        if 0 <=action<= 3:
            env_state[10+int(player_id//2)] -= level_bet[action]
            env_state[16+int(player_id//2)] += level_bet[action]
            for id in range(2):
                choice_card = weighted_random(env_state)
                env_state[37+10*player_id+choice_card] += 1
            arr_card_on_hand = env_state[(37+10*player_id):(37+10*(player_id+1))]
            point_set = Caculus_card(arr_card_on_hand)
            env_state[177+player_id] =point_set
        if action == 4:
            env_state[23+player_id] = 0
        if action == 5:
            choice_card_bs = weighted_random(env_state)
            env_state[37+10*player_id+choice_card_bs] += 1
            arr_card_on_hand = env_state[(37+10*player_id):(37+10*(player_id+1))]
            point_set = Caculus_card(arr_card_on_hand)
            env_state[177+player_id] =point_set
        if action == 6:
            env_state[10+int(player_id//2)] -= env_state[16+int(player_id//2)]
            env_state[16+int(player_id//2)] *= 2
            choice_card_bs = weighted_random(env_state)
            env_state[37+10*player_id+choice_card_bs] += 1
            arr_card_on_hand = env_state[(37+10*player_id):(37+10*(player_id+1))]
            point_set = Caculus_card(arr_card_on_hand)
            env_state[177+player_id] =point_set
            env_state[23+player_id] = 0 #end_set
        if action == 7:
            env_state[10+int(player_id//2)] -= env_state[16+int(player_id//2)]
            env_state[16+int(player_id//2)] *= 2
            ###________________________tách và chia thêm 2 lá mới cho 2 bộ__________________###
            set_card_one = env_state[(37+10*player_id):(37+10*(player_id+1))]
            place_double_card = np.where(set_card_one)[0]
            env_state[37+10*player_id+place_double_card] -= 1
            env_state[37+10*(player_id+1)+place_double_card] = 1
            for ix in range(2):
                choice_card_bs = weighted_random(env_state)
                env_state[37+10*(player_id+ix)+choice_card_bs] += 1
                arr_card_on_hand = env_state[(37+10*(player_id+ix)):(37+10*(player_id+1+ix))]
                point_set = Caculus_card(arr_card_on_hand)
                env_state[177+player_id+ix] = point_set
    if player_id == 0:
        if action == 8:
            check = 1
            if np.sum(env_state[37:47]) == 0:
                for card in range(2):
                    if card == 0:
                        choice_card_bs = weighted_random(env_state)
                        env_state[37+choice_card_bs] += 1
                        if choice_card_bs == 0:
                            env_state[177] += 11
                        if choice_card_bs != 0:
                            env_state[177] += (choice_card_bs+1)
                    if card == 1:
                        choice_card_bs = weighted_random(env_state)
                        env_state[200] = choice_card_bs 
                ###___Điểm_của_cái_chỉ_hiện_điểm_của_lá_mở___###
                if env_state[37] == 1 or env_state[200] == 0:       #check_blj_của_cái(nếu có sẽ mở lá úp)
                    upside_dow_card = np.where(env_state[37:47]==1)[0]
                    if (upside_dow_card + env_state[200]) == 9:
                        env_state[37+env_state[200]] += 1
                        env_state[177] = 21
                check -= 1
            if (np.sum(env_state[37:47]) != 0) and (check == 1):
                ##___________________check_point_____________##
                arr_card_on_hand = np.full(10,0)
                arr_card_on_hand = env_state[37:47]
                arr_card_on_hand[env_state[200]] += 1
                point_set = Caculus_card(arr_card_on_hand)
                # print("Điểm chưa update: ",point_set)
                if point_set > 16:
                    env_state[23] = 0   #end
                if point_set <= 16:
                    arr_card_on_hand[env_state[200]] -= 1
                    choice_card_bs = weighted_random(env_state)
                    env_state[37+choice_card_bs] += 1
                    arr_card_on_hand = env_state[37:47]
                    point_set = Caculus_card(arr_card_on_hand)
                    env_state[177] = point_set
    
    ###_________________________________reset_small_game___________________________###
    if (np.sum(env_state[(37+20*int(player_id//2)):(37+20*(1+int(player_id//2)))]) == 2) and (env_state[177+player_id] == 21):
        env_state[23+player_id] = 0
    
    status_player_full = env_state[23:37]
    if np.sum(status_player_full) == 0:
        ###________________________trả điểm cho Cái________________###
        # env_state[37+env_state[200]] += 1
        arr_card_on_hand = env_state[37:47]
        point_set = Caculus_card(arr_card_on_hand)
        env_state[177] = point_set
        ###________________________pay_coin______________________###
        list_id_blackjack = player_blackjack(env_state)
        list_coin_return = coin_return(list_id_blackjack, env_state)
        for win in range(6):
            if list_coin_return[1+win] > 0:
                env_state[191+win] += 1
        list_coin_return[1:7] += env_state[17:23]
        env_state[10:17] += list_coin_return    #cập nhật xong
        ###_______________________update_small_game______________###
        env_state[17:191] = 0
        env_state[197] += 1 #+1small_game
        check_player_next = np.where(env_state[11:17] >= 10)[0]
        for nextt in range(len(check_player_next)):
            env_state[25+2*check_player_next[nextt]] = 1
        env_state[23] = 1
        env_state[199] = 0
        if (env_state[197] == 30) or ((env_state[197]<30) and (np.sum(env_state[23:37])==1)):
            env_state[198] = 1
        
    if (np.sum(status_player_full) > 0) and (np.sum(env_state[37:47]) != 0):
        next_id = next_id_player(env_state)
        env_state[199] = next_id
    
    ###______________________________________________check_L____________________________________###
    # print("Lá bài còn trên bàn:", env_state[:10])
    # print("Sô tiền cược của mỗi người: ", env_state[17:23])
    # print("Số tiền còn lại của mỗi người: ", env_state[10:17])
    # print("Tình trạng được chơi:", env_state[23:37])
    # print("Bài trên tay người 0:", env_state[37:57])
    # print("Bài trên tay người 1:", env_state[57:77])
    # print("Bài trên tay người 2:", env_state[77:97])
    # print("Bài trên tay người 3:", env_state[97:117])
    # print("Bài trên tay người 4:", env_state[117:137])
    # print("Bài trên tay người 5:", env_state[137:157])
    # print("Bài trên tay người 6:", env_state[157:177])
    # print("Điểm: ", env_state[177:191])
    # print("Agent đang nhận action: ", env_state[199])
    # print("Small_game: ", env_state[197])
    # print("Lá úp cái: ", env_state[200])
    # print("________________________________________________________________")

    return env_state

@njit()
def getAgentsize():
    return 7

@njit()
def checkEnded(env_state):
    if env_state[198] == 0:
        return -1
    begin_coin = np.array([100000,1000,1000,1000,1000,1000,1000])
    curcoin = env_state[10:17] - begin_coin
    # print(curcoin)
    max_stonk = np.argmax(curcoin)
    return max_stonk

@njit()
def getReward(p_state):
    if p_state[174] == 0:
        return 0
    beg_coin = np.array([100000,1000,1000,1000,1000,1000,1000])
    cur_coin = p_state[164:171] - beg_coin
    max_stk = np.argmax(cur_coin)
    if max_stk == 0:
        return 1
    if max_stk != 0:
        return -1
    
@njit()
def getStateSize():
    return 175

def randomBot(p_state, perData):             #bot_thuong
    # print(p_state[172])
    validAction = getValidActions(p_state)
    validActions = np.where(validAction==1)[0]
    idx = np.random.choice(validActions)
    return idx, perData

@njit()
def numbaRandomBot(p_state, perData):            #bot_Numbaa
    validActions = getValidActions(p_state)
    validActions = np.where(validActions==1)[0]
    idx = np.random.choice(validActions)
    return idx, perData

def one_game(listAgent, perData):#----------------------------------------------
    env_state = initEnv()
    winner = -1
    while env_state[201] < 1000:
        pIdx = int(env_state[199]//2)
        # if turn != pIdx :
        #   turn = pIdx
        #   print("--------Turn =", env_[199])
        action, perData = listAgent[pIdx](getAgentState(env_state), perData) #kéo_action
        # print(action)
        stepEnv(action, env_state) #kéo_hàm_xử_lý_environment
        winner = checkEnded(env_state)
        if winner != -1:
            perData[0][0] += 1      #Agent_basic_2 (Tắt đi khi không test với agent này)

            break
    
    env_state[198] = 1
    if winner == 1:
        env_state[199] = 2*winner
        p_state = getAgentState(env_state)
        perData = train_Agent(p_state, perData)

    # for pIdx in range(7):         #trả_thêm_cho_mỗi_agent_1_action_để_check
    #     env_state[199] = 2*pIdx
    #     action, perData = listAgent[pIdx](getAgentState(env_state), perData)
    return winner, perData

def normal_main(listAgent, times, perData):#------------------------------------
    if len(listAgent) != 7:
        raise Exception('Hệ thống chỉ cho phép có đúng 7 người chơi!!!')
    numWin = [0, 0, 0, 0, 0, 0, 0, 0]
    pIdOrder = np.arange(7)
    for _ in range(times):
        # if printMode and _ != 0 and _ % k == 0:
        #     print(_, numWin)
        np.random.shuffle(pIdOrder)
        # print(pIdOrder)
        shuffledListAgent = [listAgent[i] for i in pIdOrder]   
        winner, perData = one_game(shuffledListAgent, perData)
        if winner == -1:
            numWin[7] += 1
        else:
            numWin[pIdOrder[winner]] += 1
    Rate_winSys_Agent = numWin[1]/times
    Rate_winDealer_Agent = numWin[1]/numWin[0]
    Rate_Compared_OtherAG = numWin[1]/(times-numWin[0])
    Rate_ChG = (times-numWin[0])/6
    print(_+1, numWin)
    print("Rate_WinSYS_Agent: ",Rate_winSys_Agent)  
    print("Rate_winDealer_Agent: ", Rate_winDealer_Agent)
    print("Rate_Compared_OtherAG :", Rate_Compared_OtherAG)
    if numWin[1] > Rate_ChG:
        print("Vượt qua các Agent khác")
    else:
        print("Chưa vượt qua các Agent khác")
    print(perData)
    return numWin, perData

def Train_Agent(listAgent, times, perData):     #Return ra file per đã train được
    if len(listAgent) != 7:
        raise Exception('Hệ thống chỉ cho phép có đúng 7 người chơi!!!')
    numWin = [0, 0, 0, 0, 0, 0, 0, 0]
    pIdOrder = np.arange(7)
    for _ in range(times):
        shuffledListAgent = [listAgent[i] for i in pIdOrder]
        winner, perData = one_game(shuffledListAgent, perData)
        if winner == -1:
            numWin[7] += 1
        else:
            numWin[pIdOrder[winner]] += 1   
    return perData


@njit()
def numba_one_game(p0, p1, p2, p3, p4, p5, p6, perData, pIdOrder):
    env_state = initEnv()

    winner = -1
    while env_state[201] < 1000:
        pIdx = int(env_state[199]//2)
        if pIdOrder[pIdx] == 0:
            action, perData = p0(getAgentState(env_state), perData)
        elif pIdOrder[pIdx] == 1:
            action, perData = p1(getAgentState(env_state), perData)
        elif pIdOrder[pIdx] == 2:
            action, perData = p2(getAgentState(env_state), perData)
        elif pIdOrder[pIdx] == 3:
            action, perData = p3(getAgentState(env_state), perData)
        elif pIdOrder[pIdx] == 4:
            action, perData = p4(getAgentState(env_state), perData)
        elif pIdOrder[pIdx] == 5:
            action, perData = p5(getAgentState(env_state), perData)
        elif pIdOrder[pIdx] == 6:
            action, perData = p6(getAgentState(env_state), perData)
        
        stepEnv(action, env_state)
        winner = checkEnded(env_state)
        if winner != -1:
            break
   
    env_state[198] = 1
    for pIdx in range(7):
        env_state[199] = 2*pIdx
        if pIdOrder[pIdx] == 0:
            action, perData = p0(getAgentState(env_state), perData)
        elif pIdOrder[pIdx] == 1:
            action, perData = p1(getAgentState(env_state), perData)
        elif pIdOrder[pIdx] == 2:
            action, perData = p2(getAgentState(env_state), perData)
        elif pIdOrder[pIdx] == 3:
            action, perData = p3(getAgentState(env_state), perData)
        elif pIdOrder[pIdx] == 4:
            action, perData = p4(getAgentState(env_state), perData)
        elif pIdOrder[pIdx] == 5:
            action, perData = p5(getAgentState(env_state), perData)
        elif pIdOrder[pIdx] == 6:
            action, perData = p6(getAgentState(env_state), perData)
    return winner, perData

@njit()
def numba_main(p0, p1, p2, p3, p4, p5, p6, times, perData):
    numWin = np.full(8, 0)
    pIdOrder = np.arange(7)
    for _ in range(times):
        # if printMode and _ != 0 and _ % k == 0:
        #   print(_, numWin)
        # np.random.shuffle(pIdOrder)
        winner, perData = numba_one_game(p0, p1, p2, p3, p4, p5, p6, perData, pIdOrder)

        if winner == -1:
            numWin[7] += 1
        else:
            numWin[pIdOrder[int(winner)]] += 1
    return numWin, perData

@jit()
def one_game_numba(p0, list_other, per_player, per1, per2, per3, per4, per5, per6, p1, p2, p3, p4, p5, p6):
    env_state = initEnv()
    while env_state[201] < 1000:
        idx = int(env_state[199]//2)
        p_state = getAgentState(env_state)
        list_action = getValidActions(p_state)

        if list_other[idx] == -1:
            action, per_player = p0(p_state,per_player)
        elif list_other[idx] == 1:
            action, per1 = p1(p_state,per1)
        elif list_other[idx] == 2:
            action, per2 = p2(p_state,per2)
        elif list_other[idx] == 3:
            action, per3 = p3(p_state,per3)
        elif list_other[idx] == 4:
            action, per3 = p3(p_state,per4)
        elif list_other[idx] == 5:
            action, per3 = p3(p_state,per5)
        elif list_other[idx] == 6:
            action, per3 = p3(p_state,per6)

        if list_action[action] != 1:
            raise Exception('Action không hợp lệ')
        stepEnv(action, env_state)
        if checkEnded(env_state) != -1:
            break

    turn = env_state[201]
    env_state[198] = 1
    for idx in range(7):
        env_state[199] = 2*idx
        if list_other[idx] == -1:
            p_state = getAgentState(env_state)
            act, per_player = p0(p_state, per_player)

    env_state[201] = turn
    winner = 0
    if np.where(list_other == -1)[0] == checkEnded(env_state): winner = 1       #check_agent_train_có_win_ko
    else: winner = 0
    return winner, per_player

@njit()
def random_Env(p_state, per):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], per

@jit()
def n_game_numba(p0,num_game, per_player, list_other, per1, per2, per3, per4, per5, per6, p1, p2, p3, p4, p5, p6):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner,per_player = one_game_numba(p0, list_other, per_player, per1, per2, per3, per4, per5, per6, p1, p2, p3, p4, p5, p6)
        win += winner
    return win, per_player

# import importlib.util, json, sys
# from setup import SHOT_PATH

# def load_module_player(player):
#    return importlib.util.spec_from_file_location('Agent_player', f"{SHOT_PATH}Agent/{player}/Agent_player.py").loader.load_module()

#@jit__()
# def numba_main_2(p0, n_game, per_player, level, *args, getAgentSize()):
#     list_other = np.array([1, 2, 3, 4, 5, 6, -1])
#     if level == 0:
#         per_agent_env = np.array([0])
#         return n_game_numba(p0, n_game, per_player, list_other, per_agent_env, per_agent_env, per_agent_env, random_Env, random_Env, random_Env)
#     else:
#         env_name = sys.argv[1]
#         if len(args) > 0:
#             dict_level = json.load(open(f'{SHOT_PATH}Log/check_system_about_level.json'))
#         else:
#             dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))

#         if str(level) not in dict_level[env_name]:
#             raise Exception('Hiện tại không có level này') 

#         lst_agent_level = dict_level[env_name][str(level)][2]
#         p1 = load_module_player(lst_agent_level[0]).Test
#         p2 = load_module_player(lst_agent_level[1]).Test
#         p3 = load_module_player(lst_agent_level[2]).Test
#         p4 = load_module_player(lst_agent_level[3]).Test
#         p5 = load_module_player(lst_agent_level[4]).Test
#         p6 = load_module_player(lst_agent_level[5]).Test
#         per_level = []
#         for id in range(getAgentSize()-1):
#             data_agent_env = list(np.load(f'{SHOT_PATH}Agent/{lst_agent_level[id]}/Data/{env_name}_{level}/Train.npy',allow_pickle=True))
#             per_level.append(data_agent_env)

#         return n_game_numba(p0, n_game, per_player, list_other, per_level[0], per_level[1], per_level[2], per_level[3], per_level[4], per_level[5], p1, p2, p3, p4, p5, p6)


###_______________________________Start_Agent_1(Test_perDATA)______________________________###

# perData = [np.zeros(3),np.zeros(getStateSize()),np.zeros(getStateSize()),np.zeros(getStateSize())]

@njit()
def agent_find_value(p_state, perData):
    actions = getValidActions(p_state)
    output = np.random.rand(getActionSize()) * actions + actions
    action = np.argmax(output)
    win = getReward(p_state)
    if perData[0][0] < 2:
        if win == 1:
            perData[1] = np.minimum(perData[1],p_state)
            perData[2] = np.maximum(perData[2],p_state)
            perData[3] = p_state
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

@njit()
def train_Agent(p_state, perData):
    if perData[0][0] < 2:
        perData[1] = np.minimum(perData[1],p_state)
        perData[2] = np.maximum(perData[2],p_state)
        perData[3] = p_state
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
    return perData

###___________________________________Agent_Basic_Action_____________________________###

perData = [np.zeros(3),np.zeros(len(getActionSize())),np.zeros(len(getActionSize()))]
@njit()
def Agent_Basic_2(p_state, perData):
    actions = getValidActions(p_state)
    if np.sum(perData[2]) == 0.0:
        Base = np.random.rand(getActionSize())
        UtAct = np.sort(Base)
        perData[2] = Base
    Can_choose = np.where(actions == 1)[0]
    for id in range(len(perData[2])):
        ValueID = UtAct[len(perData[2])-1-id]
        locations = np.where(perData[2] == ValueID)[0]
        if locations[0] in Can_choose:
            action = locations[0]

    return action, perData

@njit()
def Train_Agent_Basic_2(p_state, perData):
    if np.sum(perData[2]) == 0.0:
        Base = np.random.rand(getActionSize())
        UtAct = np.sort(Base)
        perData[2] = Base

    perData[0][0] += 1      #số ván đã chạy qua
    win = getReward(p_state)
    if win == 1:
        perData[0][1] += 1    #số ván đã thắng
    if perData[0][0] == 1000:
        perData[0][0] = 0
        if perData[0][1] > perData[0][2]:   #số ván thắng hiện tại đã nhiều hơn trước
            perData[0][2] = perData[0][1]
            perData[0][1] = 0
            perData[1] = perData[2]     #per1 lưu policy chỉ mức độ ưu tiên action đang được áp dụng
        else:
            Policy = np.random.random(getActionSize())
            perData[2] = Policy
    return perData

###_______________________________________________________________________###


listAgent = [randomBot,Agent_Basic_2,randomBot,randomBot,randomBot,randomBot,randomBot]

# normal_main(listAgent, 1, perData)
# print(numba_main(numbaRandomBot,agent_find_value,numbaRandomBot,numbaRandomBot,numbaRandomBot,numbaRandomBot,numbaRandomBot,1,perData))

# listAgent = [randomBot,randomBot,randomBot,randomBot,randomBot,randomBot,randomBot]
# perData = np.zeros(1)

normal_main(listAgent, 10000, perData)
# print(numba_main(numbaRandomBot,agent_find_value,numbaRandomBot,numbaRandomBot,numbaRandomBot,numbaRandomBot,numbaRandomBot,1,perData))
