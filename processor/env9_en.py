import json
import random
import numpy as np
from collections import Counter
import string
from copy import deepcopy
import logging
import platform
import time
from difflib import SequenceMatcher


class WereWolf9:
    def __init__(self, dump_log=False, debug=False, max_time=14):
        self.max_time = max_time
        self.debug = debug

        self.role_config = [
            {
                'role': 'villager',
                'role_id': 0,
                'number': 3,
                'camp': 'villager',
                'camp_id': 0,
                'open_eye_sequence': -1,
            },
            {
                'role': 'werewolf',
                'role_id': 1,
                'number': 3,
                'camp': 'werewolf',
                'camp_id': 1,
                'open_eye_sequence': 100,
            },
            {
                'role': 'seer',
                'role_id': 2,
                'number': 1,
                'camp': 'special',
                'camp_id': 2,
                'open_eye_sequence': 12,
                'seen': [],  # players that have been checked
            },
            {
                'role': 'witch',
                'role_id': 3,
                'number': 1,
                'camp': 'special',
                'camp_id': 2,
                'open_eye_sequence': 13,
                'remain_save': 1,
                'remain_poison': 1,
            },
            {
                'role': 'hunter',
                'role_id': 4,
                'number': 1,
                'camp': 'special',
                'camp_id': 2,
                'open_eye_sequence': 11,
            },
        ]

        self.shared_init_attributes = {
            'status': 'in_game',
            'is_revealed': False,  # if the identity is public to all players
            'action_history': [],
            'speak_history': [],
        }

        self.task_id_to_name = {
            # night tasks
            100: 'Werewolf vote for kill',  # 狼人第一轮投票
            101: 'Werewolf vote for kill, the second vote',  # 狼人第二轮投票
            102: 'Werewolf vote for kill, the last vote',  # 狼人第三轮投票

            110: 'Witch antidote',  # 女巫解药
            111: 'Witch poison',  # 女巫毒药

            120: 'Seer check',  # 预言家查验身份
            130: 'Hunter shoot at night',  # 猎人晚上被杀开枪

            # day tasks
            200: 'Last words for died at night',  # 被杀遗言
            210: 'First round of speeches',  # 第一轮发言
            220: 'First round of vote',  # 第一轮投票
            230: 'Second round of speeches',  # 第二轮发言
            240: 'Second round of vote',  # 第二轮投票
            250: 'Last words for been exiled',  # 放逐遗言
            260: 'Hunter shoot at night',  # 猎人白天开枪

            270: 'Werewolf commit suicide',  # 狼人自爆
            280: 'Last words for been shot',  # 枪杀遗言
        }

        # 存活类型
        self.status_eng = {
            'in_game': '存活',  # normal status, public
            'to_be_killed': '即将被杀',  # to be killed by werewolves, only visible for the Witch
            'killed': '狼杀',  # killed by werewolves, only visible for werewolves
            'saved': '被救',  # killed by werewolves and saved by the Witch, only visible for werewolves and the Witch
            'poisoned': '毒杀',  # poisoned by the Witch, only visible for the Witch
            'shot_day': '枪杀',  # shot by the Hunter, public
            'exiled': '票杀',  # exiled after day voting, public
            'dead': '死亡',  # died at night, might be killed or poisoned, public
            'suicide': '自爆',  # werewolf commit suicide, public
        }

        self.player_num = 9
        self.time = 0
        self.num_step = 0
        self.attributes = {}
        self.round_task = {}
        self.round_action = {}

        self.public_role = {}
        self.public_action = {}
        self.public_speak = {}

        self.game_result = -1
        self.out_game_history = {}

        self.player_to_be_killed = -1
        self.player_to_be_saved = -1
        self.player_to_be_poisoned = -1
        self.player_to_be_exiled = -1
        self.player_to_be_shoot = -1
        self.player_to_be_suicide = -1

        self.speak_second_round = []
        self.speak_order = []
        self.init_logger()
        self.speak_history = []

        self.dump_log = dump_log
        if self.dump_log:
            self.game_log = {}

        self.random_vote = False
        self.suicide_flag = False

    def init_logger(self):
        if self.debug:
            logging_level = logging.INFO
        else:
            logging_level = logging.ERROR

        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        self.logger = logging.getLogger('env')
        self.logger.setLevel(logging_level)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def set_first_night(self, wolf_to_seer=-1, see_id=-1, save_id=-1, kill_id=-1):
        wolves = [pid for pid in range(9) if self.roles[pid] == "werewolf"]
        if wolf_to_seer == -1:
            wolf_to_seer = random.choice([0, 1, 2, -1])
        if wolf_to_seer in {0, 1, 2}:
            self.wolf_to_seer = wolves[wolf_to_seer]
        else:
            self.wolf_to_seer = -1
        if see_id == -1 and self.speak_seq is None:
            see_id = random.choice(list(range(8)))
            others = [pid for pid in range(9) if self.roles[pid] != "seer"]
            self.see_id = others[see_id]
        else:
            self.see_id = see_id
        self.fake_list[self.roles.index('seer')] = self.see_id

        if save_id == -1:
            save_id = random.choice([0, 1])
        self.save_id = save_id

        if kill_id == -1:
            kill_id = random.choice(list(range(9)))
        self.kill_id = kill_id

        return

    def mask_strategy(self, player_id):
        if player_id == -1:
            return self.fake_list
        strategy = [-1] * 9

        for i in range(len(self.fake_list)):
            # 自己看的到自己的策略，狼人互相能看到策略
            if self.roles[player_id] == self.roles[i] and self.roles[i] != 'villager':
                strategy[i] = self.fake_list[i]

        return strategy

    def get_info(self, player_id):
        info = {}
        info["roles"] = self.roles
        info["random_vote"] = self.random_vote
        info["wolf_order"] = self.wolf_order
        info["speak_order"] = self.speak_order
        info["speak_seq"] = self.speak_seq

        info["werewolf_kill_history"] = self.werewolf_kill_history
        info["witch_save_history"] = self.witch_save_history
        info["witch_poison_history"] = self.witch_poison_history
        info["seer_check_history"] = self.seer_check_history

        return info

    def reset(self, role_list=None, speak_seq=None, allow_suicide=False, seed=None):
        if seed is None:
            self.game_id = random.getrandbits(32)
        else:
            self.game_id = seed
        np.random.seed(self.game_id)
        random.seed(self.game_id)

        self.time = 0
        self.num_step = 0
        self.game_result = -1
        self.out_game_history = {}
        self.speak_order = []
        self.speak_history = []
        self.allow_suicide = allow_suicide

        self.seer_check_history = []
        self.werewolf_kill_history = []
        self.witch_save_history = []
        self.witch_poison_history = []

        # 初始化所有角色
        player_list = []
        for role in self.role_config:
            num = role['number']
            for i in range(num):
                player = deepcopy(role)
                player.update(deepcopy(self.shared_init_attributes))
                player_list.append(player)

        # 映射玩家和角色
        if role_list is None:
            random.shuffle(player_list)
        else:
            # 创建一个临时字典来存储原始列表中的对象和它们对应的role
            temp_dict = {}
            for obj in player_list:
                role = obj['role']
                if role not in temp_dict:
                    temp_dict[role] = [obj]
                else:
                    temp_dict[role].append(obj)
            # 根据新顺序列表重新排序原始列表
            for i, role in enumerate(role_list):
                player_list[i] = temp_dict[role].pop(0)

        for player_id in range(len(player_list)):
            self.attributes[player_id] = {'player_id': player_id}
            self.attributes[player_id].update(player_list[player_id])
        self.roles = [self.attributes[kk]["role"] for kk in range(9)]

        self.speak_seq = speak_seq

        self.logger.info(
            '=======================================================================================================')
        self.logger.info('Game start, id %d' % self.game_id)
        self.gen_task()

        player_id, state, legal_action = self.get_state_and_legal_action()
        self.random_vote = False

        reward = 0.
        done = False
        info = self.get_info(player_id)

        return player_id, state, legal_action, reward, done, info

    def get_legal_action(self, task_id, player_id):
        legal_action = self.get_in_game_player_id()

        # 狼人指定杀人
        if task_id in [100, 101, 102]:
            # fanlang允许狼人弃刀
            legal_action.append(-1)

        # fanlang里面女巫所有夜晚都可以对自己用技能, 但是是否生效由gamecore判断

        # 女巫解药
        elif task_id == 110:
            assert self.attributes[player_id]['remain_save'] > 0  # 没有解药理论上不会有这个回合
            legal_action = [-1]
            if self.player_to_be_killed > -1:
                legal_action.append(self.player_to_be_killed)

        # 女巫毒药
        elif task_id == 111:
            assert self.player_to_be_saved == -1  # 使用过解药理论上不会有这个回合
            assert self.attributes[player_id]['remain_poison'] > 0  # 没有毒药理论上不会有这个回合
            legal_action.append(-1)

        # 预言家查验身份
        elif task_id == 120:
            # 不能查以前查过的
            seen = self.attributes[player_id]['seen']
            # 可以查已经出局的
            unseen = sorted(list(set(self.get_all_player_id()) - set(seen)))
            # 不能查自己
            if player_id not in unseen:
                assert False
            unseen.remove(player_id)
            # 不能查身份已经揭晓的
            for tmp_id in unseen:
                if self.attributes[tmp_id]['is_revealed'] is True:
                    unseen.remove(tmp_id)
            assert len(unseen) > 0
            legal_action = unseen

            # 可以不查
            legal_action.append(-1)

        # 猎人开枪
        elif task_id == 260:
            if self.attributes[player_id]['role'] == 'hunter' and \
                    len(self.get_in_game_player_id(role_ids=[2, 3, 4])) > 0:
                # 猎人不能对自己开枪
                if player_id in legal_action:
                    legal_action.remove(player_id)
                legal_action.append(-1)
            else:
                assert False

        # 被杀遗言
        elif task_id == 200:
            legal_action = [self.player_num]

        # 第一轮发言
        elif task_id == 210:
            legal_action = [self.player_num]

        # 第一轮投票
        elif task_id == 220:
            # 允许投自己, 允许弃权
            # legal_action.remove(player_id)
            legal_action.append(-1)

        # 第二轮发言
        elif task_id == 230:
            legal_action = [self.player_num]

        # 第二轮投票
        elif task_id == 240:
            legal_action = self.speak_second_round
            legal_action.append(-1)

        # 放逐遗言
        elif task_id == 250:
            legal_action = [self.player_num]
        # 狼人自爆
        elif task_id == 270:
            legal_action = [player_id, -1]
        elif task_id == 280:
            legal_action = [self.player_num]


        else:
            assert False

        return legal_action

    def get_game_obs(self, done=False):
        # 当前时间和任务
        if done:
            game_obs = {
                'time': self.time,
                'task_name': '游戏结束'
            }
            return game_obs
        task_id = self.get_task_id()
        task_name = self.task_id_to_name[task_id]

        game_obs = {
            'time': self.time,
            'task_name': task_name,
        }

        return game_obs

    def get_role_obs(self, id_self, done=False):
        '''
            1. 已知的玩家身份
            (1) 自己的身份: 自己可知
            (2) 狼人互相知道身份: 仅狼人内部可知
            (3) 预言家历史查验的身份: 仅预言家可知
            (4) 猎人死亡时开枪暴露身份: 全体可知
        '''
        if done:
            player_ids = self.get_all_player_id()
            role_obs = []
            for player_id in player_ids:
                role = self.attributes[player_id]['role']
                role_obs.append(role)
            return role_obs

        role_self = self.attributes[id_self]['role']
        player_ids = self.get_all_player_id()

        role_obs = []
        for player_id in player_ids:
            role = self.attributes[player_id]['role']
            is_revealed = self.attributes[player_id]['is_revealed']

            # 自己知道自己的身份
            if player_id == id_self:
                is_revealed = True

            # 狼人互相知道身份, 仅狼人内部可知
            elif role_self == 'werewolf':
                if role == 'werewolf':
                    is_revealed = True

            # 预言家历史查验的身份：仅预言家可知
            elif role_self == 'seer':
                seen = self.attributes[id_self]['seen']
                if player_id in seen:
                    if not is_revealed:
                        is_revealed = 'checked'
                    else:
                        # 身份揭晓比查验信息量更大
                        pass

            if is_revealed == 'checked':
                if role == 'werewolf':
                    role_obs.append(role)
                else:
                    role_obs.append('goodman')

            # 猎人死亡时开枪暴露身份：全体可知
            # 白痴被放逐后暴露身份：全体可知
            elif is_revealed is True:
                role_obs.append(role)
            elif is_revealed is False:
                role_obs.append('unknown')
            else:
                assert False

        return role_obs

    def get_status_obs(self, id_self, done=False):
        # 女巫毒药和解药的数量：仅女巫可知，女巫可以从自己的历史action中得出（非-1的数量），无需额外给出

        if not done:
            role_self = self.attributes[id_self]['role']
        status_obs_all = []
        player_ids = self.get_all_player_id()
        if done:
            for player_id in player_ids:
                status_obs = self.attributes[player_id]['status']
                status_obs_all.append(status_obs)
            return status_obs_all

        for player_id in player_ids:
            status_obs = self.attributes[player_id]['status']

            # 只有狼人和女巫知道被刀信息
            # 当女巫没有药之后, 虽然获取了被刀信息但是没有女巫回合, 此时白天公布死亡一定是狼人刀死, 所以没有信息泄露
            if status_obs == 'killed' and role_self not in ['werewolf', 'witch']:
                status_obs = 'dead'

            # 被狼人刀, 但还未执行, 狼人和女巫(女巫有解药)可见
            if status_obs == 'to_be_killed' and role_self not in ['werewolf', 'witch']:
                status_obs = 'in_game'
            if status_obs == 'to_be_killed' and role_self == "witch" and self.attributes[id_self]['remain_save'] == 0:
                status_obs = "in_game"
            # 被狼人刀的猎人自己可见
            if status_obs == 'to_be_killed' and player_id == id_self and role_self == 'hunter':
                status_obs = 'to_be_killed'

            # 女巫知道中毒信息, 狼人知道解药
            if status_obs == 'poisoned' and role_self not in ['witch', 'werewolf']:
                status_obs = 'dead'
            if status_obs == 'saved' and role_self not in ['witch', 'werewolf']:
                status_obs = 'in_game'
            status_obs_all.append(status_obs)

        return status_obs_all

    def get_out_game_obs(self, id_self):
        # 此信息作为非狼人和女巫之外人的补充信息, 即需要记录历史出局的时间
        return self.out_game_history

    def get_action_obs(self, id_self, done=False):
        '''
         历史动作
        （1）狼人投票指定杀人：仅狼人内部可知
        （2）预言家历史查验动作：仅预言家可知，但是可以通过legal action 屏蔽已经查验的对象
        （3）狼人每晚投票决定的杀人对象：仅女巫可知
        （4）女巫历史使用毒药和解药的记录：仅女巫可知
        （5）所有玩家白天的投票，但不包括当前轮次：全体可知
        '''

        if not done:
            role_self = self.attributes[id_self]['role']
        player_ids = self.get_all_player_id()

        action_all = {}
        for player_id in player_ids:
            action_history = self.attributes[player_id]['action_history']
            for info in action_history:
                time, task_id, action = info

                # 非狼人看不到狼人投票信息
                if not done:
                    if task_id in [100, 101, 102] and role_self != 'werewolf': continue

                    if task_id in [110, 111] and role_self != 'witch': continue

                    if task_id == 120 and role_self != 'seer': continue

                    if task_id == 260 and action == -1: continue
                    # 狼人自爆
                    if task_id == 270 and action == -1 and role_self != "werewolf": continue

                # 猎人晚上开枪由于白天会翻牌, 所以认为可知
                task_name = self.task_id_to_name[task_id]
                if time not in action_all.keys():
                    action_all[time] = {}
                if task_name not in action_all[time].keys():
                    action_all[time][task_name] = []
                action_all[time][task_name].append([player_id, action])

        # 白天投票的时候看不到当前回合其他玩家的投票信息
        if not done:
            if self.time in action_all.keys():
                if list(self.round_task.keys())[0] == 220:
                    if '第一轮投票' in action_all[self.time].keys():
                        del action_all[self.time]['第一轮投票']
                if list(self.round_task.keys())[0] == 240:
                    if '第二轮投票' in action_all[self.time].keys():
                        del action_all[self.time]['第二轮投票']

        # TODO task_name 顺序由player_id决定而不是task_id顺序决定, 不太合理
        return action_all

    def get_speak_obs(self):
        # 所有玩家白天的发言：全体可知
        player_ids = self.get_all_player_id()
        speak_all = {}

        # for player_id in player_ids:
        #     speak_history = self.attributes[player_id]['speak_history']
        #     for info in speak_history:
        #         time, task_id, speak = info
        #         task_name = self.task_id_to_name[task_id]
        #         if time not in speak_all.keys():
        #             speak_all[time] = {}
        #         if task_name not in speak_all[time].keys():
        #             speak_all[time][task_name] = []
        #         speak_all[time][task_name].append([player_id, speak])

        for info in self.speak_history:
            time, task_id, speak, player_id = info
            task_name = self.task_id_to_name[task_id]
            if time not in speak_all.keys():
                speak_all[time] = {}
            if task_name not in speak_all[time].keys():
                speak_all[time][task_name] = []
            speak_all[time][task_name].append([player_id, speak])
        # TODO task_name 顺序由player_id决定而不是task_id顺序决定, 不太合理

        return speak_all

    def get_speak_history(self):
        # 所有玩家白天的发言：全体可知
        player_ids = self.get_all_player_id()
        speak_all = {}

        for player_id in player_ids:
            speak_history = self.attributes[player_id]['speak_history']
            speak_all[player_id] = speak_history

        return speak_all

    def get_state(self, player_id, done=False):
        # 当前对局阶段信息
        game_obs = self.get_game_obs(done)

        # 玩家身份信息
        role_obs = self.get_role_obs(player_id, done)

        # 玩家状态信息
        status_obs = self.get_status_obs(player_id, done)

        # 历史出局信息
        out_game_obs = self.get_out_game_obs(player_id)

        # 玩家动作信息
        action_obs = self.get_action_obs(player_id, done)

        # 发言信息
        speak_obs = self.get_speak_obs()
        speak_history = self.get_speak_history()
        state = {
            'game': game_obs,
            'role': role_obs,
            'status': status_obs,
            'out_game': out_game_obs,
            'action': action_obs,
            'speak': speak_obs,
            "speak_history": speak_history
        }

        return state

    def get_state_and_legal_action(self):
        if len(list(self.round_task.keys())) == 0:
            a = 1
        task_id = self.get_task_id()
        player_id = self.round_task[task_id][0]

        state = self.get_state(player_id)
        legal_action = self.get_legal_action(task_id, player_id)

        self.legal_action = legal_action
        self.legal_action_plus1 = [tmp+1 for tmp in legal_action]
        return player_id, state, legal_action

    def get_all_player_id(self):
        return list(self.attributes.keys())

    def get_in_game_player_id(self, role_ids=None):
        player_ids = []
        for pid in range(self.player_num):
            attribute = self.attributes[pid]
            if role_ids is not None and attribute['role_id'] not in role_ids: continue
            if attribute['status'] in ['in_game', 'to_be_killed', 'saved']:
                player_ids.append(pid)
        return player_ids

    def get_in_game_wolf(self, role_ids=None):
        player_ids = self.get_in_game_player_id(role_ids={1})
        return player_ids

    def get_vote_player_id(self):
        # 获取场上能够投票玩家的id
        player_ids = []
        for attribute in self.attributes.values():
            if attribute['status'] != 'in_game' and attribute['status'] != 'saved': continue

            player_ids.append(attribute['player_id'])
        return player_ids

    def get_task_at_night(self):
        self.player_to_be_killed = -1
        self.player_to_be_saved = -1
        self.player_to_be_poisoned = -1

        self.logger.info(
            '=======================================================================================================')
        self.logger.info('Night %d:' % (self.time // 2 + 1))

        # 计算睁眼顺序
        open_weight = []
        for player_id in self.attributes.keys():
            status = self.attributes[player_id]['status']
            if status in ['in_game', 'saved']:
                weight = self.attributes[player_id]['open_eye_sequence']
            else:
                weight = -1
            open_weight.append(weight)
        open_weight_set = sorted(set(open_weight))[::-1]

        round_task = {}
        for weight in open_weight_set:
            if weight < 0: break
            player_ids = np.where(np.array(open_weight) == weight)[0].tolist()

            role = self.attributes[player_ids[0]]['role']
            if role == 'werewolf':
                # 如果只剩一只狼, 只需一轮投票
                if len(player_ids) == 1:
                    task_ids = [100]
                else:
                    # task_ids = [100, 101, 102]
                    task_ids = [100]  # 取消三轮投票
                    random.shuffle(player_ids)
                self.wolf_order = player_ids[:]
            elif role == 'witch':
                task_ids = []
                remain_save = self.attributes[player_ids[0]]['remain_save']
                if remain_save > 0:
                    task_ids.append(110)
                remain_poison = self.attributes[player_ids[0]]['remain_poison']
                if remain_poison > 0:
                    task_ids.append(111)
            elif role == 'seer':
                task_ids = [120]
            elif role == "hunter":
                continue
            else:
                self.logger.error('%d, %s' % (player_ids, role))
                assert False
            for task_id in task_ids:
                round_task[task_id] = player_ids[:]
        return round_task

    def get_ordered_player(self, round=1, begin=None, order=None, legal_players=None):

        time_str = 'Day %d' % (self.time // 2 + 1)
        time_str += ' Daytime' if self.time % 2 == 1 else ' Night'
        if self.speak_seq is not None:
            if round == 1:
                round_key = 'First round of speeches'
            elif round == 2:
                round_key = 'Second round of speeches'
            else:
                assert False
            speak_list = self.speak_seq[time_str][round_key]
            speak_list = (np.array(speak_list) - 1).tolist()
            if len(speak_list) > 0:
                begin = speak_list[0]
            orders = [0, 1]
        else:
            speak_list = None
            if order is None:
                orders = [random.choice([0, 1])]
            else:
                orders = [order]

        order_player_lists = []
        for order in orders:
            last_night = self.time - 1
            out_player_last_night = [p for p in self.out_game_history[last_night] if p != -1]
            if len(out_player_last_night) == 0:
                # 无人死亡的情况下，随机序号发言
                out_player_last_night = [random.choice(self.get_in_game_player_id())]
            # 有多死的情况下,需要随机死左右发言
            for min_die_player in out_player_last_night:
                direction = order * 2 - 1
                order_player_list = []
                for i in range(self.player_num):
                    pid = (min_die_player + self.player_num + direction * i) % self.player_num
                    attribute = self.attributes[pid]
                    if attribute['status'] in {'in_game', 'to_be_killed', 'saved'}:
                        if legal_players is not None and pid not in legal_players:
                            pass
                        elif self.speak_seq is not None and pid == self.speak_seq[time_str]['Hunter shoot at night'] - 1:
                            pass
                        else:
                            order_player_list.append(pid)

                if begin is not None:
                    split_id = order_player_list.index(begin)
                    order_player_list = order_player_list[split_id:] + order_player_list[:split_id]

                order_player_lists.append(order_player_list)

        if speak_list is None:
            final_order = random.choice(order_player_lists)
        else:
            match_lens = []
            for order_player_list in order_player_lists:
                if len(order_player_list) < len(speak_list):
                    print(f'order_player_list: {order_player_list}')
                    print(f'speak_list: {speak_list}')
                    assert False
                match_len = SequenceMatcher(None, speak_list, order_player_list).find_longest_match(alo=0, ahi=len(speak_list), blo=0, bhi=len(order_player_list)).size
                match_lens.append(match_len)

            max_len_index = np.argmax(match_lens)
            max_len = match_lens[max_len_index]

            final_order = order_player_lists[max_len_index]

        return final_order

    def get_task_at_day(self):
        self.player_to_be_exiled = -1
        self.player_to_be_shoot = -1
        self.player_to_be_suicide = -1

        self.logger.info(
            '=======================================================================================================')
        self.logger.info('Day %d:' % (self.time // 2 + 1))

        round_task = {}

        # '被杀遗言'
        if self.time == 1:
            last_word_player = []
            if self.player_to_be_killed > -1:
                last_word_player.append(self.player_to_be_killed)
            if self.player_to_be_poisoned > -1:
                last_word_player.append(self.player_to_be_poisoned)
            last_word_player = sorted(last_word_player)
            if len(last_word_player) > 0:
                round_task[200] = last_word_player

        if self.player_to_be_killed > -1 and self.attributes[self.player_to_be_killed]['role'] == 'hunter':
            round_task[260] = [self.player_to_be_killed]

        word_player = self.get_in_game_player_id()

        # word_player = sorted(word_player)
        wolf_player = self.get_in_game_wolf()

        round_task[210] = self.get_ordered_player(round=1)
        self.speak_order = round_task[210][:]
        round_task[220] = self.get_vote_player_id()
        round_task[230] = []
        round_task[240] = []
        round_task[250] = []

        # 添加初始发言的狼人自爆环节
        if self.allow_suicide is True:
            round_task[270] = wolf_player[:]

        return round_task

    def print_current_task(self, task_id=None):
        if task_id is not None:
            task_name = self.task_id_to_name[task_id]
            if 'speeches' in task_name:
                task_name += ', speak order: ' + ','.join(str(id) for id in self.speak_order)
            self.logger.info('=== Task: %s ===' % task_name)
        else:
            task_ids = list(self.round_task.keys())
            if len(task_ids) > 0:
                task_id = self.get_task_id()
                self.logger.info('=== Task: %s ===' % self.task_id_to_name[task_id])
        return

    def get_task_id(self):
        task_ids = list(self.round_task.keys())
        if len(task_ids) == 0:
            return None
        task_id = task_ids[0]

        if task_id in [210, 230]:
            self.suicide_flag = True
        if self.suicide_flag:
            if 270 in task_ids:
                task_id = 270
            else:
                task_id = task_ids[0]
        else:
            task_id = task_ids[0]
        return task_id

    def gen_task(self):
        if self.time % 2 == 0:
            round_task = self.get_task_at_night()
        else:
            round_task = self.get_task_at_day()

        self.round_task = round_task
        self.round_action = {}
        task_id = self.get_task_id()

        self.print_current_task(task_id)
        return

    def vote_kill(self, vote_list):
        c = Counter(vote_list)
        tmp = c.most_common()

        count = tmp[0][1]
        ids = []
        for i in range(len(tmp)):
            if tmp[i][1] == count:
                ids.append(tmp[i][0])

        ids = sorted(ids)
        if len(ids) == 1:
            result = ids[0]
        else:
            # 狼人指定杀人必须有一个, 所以三轮投票没有一个最高就随机
            result = np.random.choice(ids)
            self.logger.info('Random vote in: %s, result: %s' % ([id+1 for id in ids], result+1))
            self.random_vote = True
        result = int(result)
        return result

    def vote_exile(self, vote_list):
        # 删除弃权票
        vote_list = [tmp for tmp in vote_list if tmp != -1]
        if len(vote_list) == 0:
            return []

        c = Counter(vote_list)
        tmp = c.most_common()

        count = tmp[0][1]
        ids = []
        for i in range(len(tmp)):
            if tmp[i][1] == count:
                ids.append(tmp[i][0])
        ids = sorted(ids)
        return ids

    def get_game_result(self, print_info=True):
        # -1 进行中 0 狼人全死 1 神全死 2 村民全死
        in_game_players = self.get_in_game_player_id()
        if print_info:
            in_game_players_print = [id+1 for id in in_game_players]
            self.logger.info('In game players: %s' % in_game_players_print)

        in_game_camps = []
        for player_id in in_game_players:
            in_game_camps.append(self.attributes[player_id]['camp'])
        in_game_camps = list(set(in_game_camps))
        if self.time >= self.max_time:
            info = '游戏超时, 好人获胜'
            game_result = 0
            if print_info:
                self.logger.info(info)
        elif len(in_game_camps) == 3:
            game_result = -1
        else:
            if 'werewolf' not in in_game_camps:
                info = 'Game end, the good camp win'
                game_result = 0
            elif 'special' not in in_game_camps:
                info = 'Game end, werewolves win, all special roles are out of the game'
                game_result = 1
            elif 'villager' not in in_game_camps:
                info = 'Game end, werewolves win, all villagers are out of the game'
                game_result = 2
            else:
                assert False
            if print_info:
                self.logger.info(info)
        return game_result

    def append_action(self, player_id, task_id, action):
        # action_history记录player所有回合的action
        if isinstance(action, str) or isinstance(action, list):
            self.attributes[player_id]['speak_history'].append([self.time, task_id, action])
            self.speak_history.append([self.time, task_id, action, player_id])
        else:
            action = int(action)
            self.attributes[player_id]['action_history'].append([self.time, task_id, action])

        # self.round_action缓存一个回合的action用于流程判断
        self.round_action[task_id].append(action)

        return

    def night(self, action):
        # night流程
        task_id = list(self.round_task.keys())[0]
        task_player_ids = self.round_task[task_id]

        if task_id not in self.round_action.keys():
            self.round_action[task_id] = []

        player_id = task_player_ids.pop(0)
        role = self.attributes[player_id]['role']
        self.logger.info('Player: %d, Role: %s, Action: %s' % (player_id + 1, role, action + 1))
        self.append_action(player_id, task_id, action)

        if len(task_player_ids) == 0:
            self.round_task.pop(task_id)
            task_ids = list(self.round_task.keys())
            if len(task_ids) > 0:
                next_task_id = task_ids[0]
            else:
                next_task_id = None

            # 狼人投票达成一致，提前结束投票回合
            kill_votes_last = list(self.round_action.values())[-1]
            if role == 'werewolf' and len(set(kill_votes_last)) == 1:
                while next_task_id in [100, 101, 102]:
                    self.round_task.pop(next_task_id)
                    task_ids = list(self.round_task.keys())
                    if len(task_ids) > 0:
                        next_task_id = task_ids[0]
                    else:
                        next_task_id = None

            # 狼人三轮投票完成
            if role == 'werewolf' and next_task_id not in [100, 101, 102]:
                # 计算准备被狼人杀死的人, 便于女巫决策
                player_to_be_killed = self.vote_kill(kill_votes_last)
                if player_to_be_killed > -1:
                    self.logger.info(
                        'Werewolf kill: Player %d, Role: %s' % (
                            player_to_be_killed + 1, self.attributes[player_to_be_killed]['role']))
                    if player_to_be_killed not in self.get_in_game_player_id():
                        assert False
                    self.attributes[player_to_be_killed]['status'] = 'to_be_killed'
                self.player_to_be_killed = player_to_be_killed
                self.werewolf_kill_history.append(player_to_be_killed)

            if role == 'witch':
                if action == -1:
                    pass
                else:
                    if task_id == 110:
                        assert action == self.player_to_be_killed

                        if player_id == self.player_to_be_killed and self.time != 0:
                            self.logger.info('女巫自救失败(只有第一晚能自救)')
                            # fanlang中算作毒杀
                            self.player_to_be_killed = -1
                            self.player_to_be_poisoned = action
                        else:
                            self.logger.info('Witch saved Player %d' % (action + 1))
                            self.attributes[self.player_to_be_killed]['status'] = 'saved'
                            self.player_to_be_killed = -1
                            self.player_to_be_saved = action
                            self.attributes[player_id]['remain_save'] -= 1

                            self.witch_save_history.append(self.player_to_be_saved)

                        if 111 in self.round_task:
                            self.round_task.pop(111)

                    elif task_id == 111:
                        self.player_to_be_poisoned = action
                        self.logger.info('Witch poison the Player %d', (action + 1))
                        self.attributes[player_id]['remain_poison'] -= 1
                        if self.player_to_be_poisoned == self.player_to_be_killed:
                            # 同时被毒和被杀  算作被毒
                            self.player_to_be_killed = -1
                        self.witch_poison_history.append(self.player_to_be_poisoned)
                    else:
                        assert False

            if role == 'seer':
                seen = self.attributes[player_id]['seen']
                assert action not in seen

                if action != -1:
                    seen.append(action)

                # 预言家在知道所有人身份后不需要睁眼 (极端情况)
                unseen = sorted(list(set(self.get_all_player_id()) - set(seen)))
                unseen.remove(player_id)
                for tmp_id in unseen:
                    if self.attributes[tmp_id]['is_revealed'] is True:
                        unseen.remove(tmp_id)
                if len(unseen) == 0:
                    self.attributes[player_id]['open_eye_sequence'] = -1

                self.seer_check_history.append(action)

            self.print_current_task()

        if len(self.round_task) == 0:
            # 计算死亡情况
            if self.player_to_be_killed > -1:
                self.logger.info('Night summary: Player %d is killed by werewolves' % (self.player_to_be_killed + 1))
                self.attributes[self.player_to_be_killed]['status'] = 'killed'
            if self.player_to_be_poisoned > -1:
                self.logger.info('Night summary: Player %d is poisoned by the Witch' % (self.player_to_be_poisoned + 1))
                self.attributes[self.player_to_be_poisoned]['status'] = 'poisoned'

            dead_player = [i for i in [self.player_to_be_killed, self.player_to_be_poisoned] if i != -1]
            if len(dead_player) == 0:
                dead_player = [-1]
            self.out_game_history[self.time] = dead_player

            self.game_result = self.get_game_result()
            if self.game_result == -1:
                self.time += 1
                self.gen_task()
        return

    def day(self, action):
        # day流程

        task_id = self.get_task_id()
        if task_id in [220, 240]:
            self.suicide_flag = False

        task_player_ids = self.round_task[task_id]

        if task_id not in self.round_action.keys():
            self.round_action[task_id] = []

        player_id = task_player_ids.pop(0)
        role = self.attributes[player_id]['role']
        if task_id != 270:
            if 9 in self.legal_action:
                self.logger.info('Player: %d, Role: %s, Speech: %s' % (player_id + 1, role, action))
            else:
                self.logger.info('Player: %d, Role: %s, Action: %s' % (player_id + 1, role, action + 1))
        self.append_action(player_id, task_id, action)

        if len(task_player_ids) == 0:
            self.round_task.pop(task_id)

            # 第一轮归票
            if task_id == 220:
                exile_votes = self.round_action[task_id]
                results = self.vote_exile(exile_votes)

                # 满足条件才会进入第二回合发言和投票: 票数并列人数 > 1, 当所有投票人的都并列top1时, 没有第二轮投票
                if 1 < len(results) < len(exile_votes):
                    second_vote_ids = sorted(list(set(self.get_vote_player_id()) - set(results)))

                    self.logger.info('Tied players: %s' % [id + 1 for id in results])
                    # 当有投票人的时候
                    if len(second_vote_ids) == 0:
                        # 当所有投票人的都并列top1时, 第二轮投票都是投票人
                        second_vote_ids = sorted(self.get_vote_player_id())

                    if self.speak_seq is not None:
                        self.speak_second_round = self.get_ordered_player(round=2, legal_players=results)
                    else:
                        pk_index_list = list(range(len(results)))
                        first_index = random.choice(pk_index_list)
                        self.speak_second_round = results[first_index:] + results[:first_index]

                    self.round_task[230] = self.speak_second_round[:]
                    self.round_task[240] = second_vote_ids

                else:
                    self.round_task.pop(230)
                    self.round_task.pop(240)
                    if len(results) == 1:
                        self.player_to_be_exiled = results[0]

            # 第二轮归票
            elif task_id == 240:
                exile_votes = self.round_action[task_id]
                results = self.vote_exile(exile_votes)
                if len(results) == 1:
                    self.player_to_be_exiled = results[0]

            if len(self.round_task) > 0 and list(self.round_task.keys())[0] == 250:
                # 计算放逐情况
                if self.player_to_be_exiled > -1:
                    self.logger.info('Player %d is exiled' % (self.player_to_be_exiled + 1))

                    if self.attributes[self.player_to_be_exiled]['role'] == 'hunter':
                        self.attributes[self.player_to_be_exiled]['status'] = 'exiled'
                        if len(self.get_in_game_player_id(role_ids=[2, 3, 4])) == 0:
                            # 游戏结束, 无需遗言和开枪
                            self.round_task.pop(250)
                        else:
                            self.round_task[250].append(self.player_to_be_exiled)
                            self.round_task[260] = [self.player_to_be_exiled]
                    else:
                        self.attributes[self.player_to_be_exiled]['status'] = 'exiled'
                        game_result_tmp = self.get_game_result(print_info=False)
                        if game_result_tmp != -1:
                            # 游戏结束, 无需遗言和开枪
                            self.round_task.pop(250)
                        else:
                            self.round_task[250].append(self.player_to_be_exiled)

                else:
                    self.round_task.pop(250)

            if task_id == 260:
                if action == -1:
                    pass
                else:
                    self.logger.info('The hunter shoot Player %d' % (action + 1))
                    self.player_to_be_shoot = action
                    self.attributes[self.player_to_be_shoot]['status'] = 'shot_day'
                    # 猎人身份揭晓
                    self.attributes[player_id]['is_revealed'] = True
                    if 210 in self.round_task:
                        if action in self.round_task[210]:
                            self.round_task[210].remove(action)
                        if action in self.round_task[220]:
                            self.round_task[220].remove(action)
                        if action in self.speak_order:
                            self.speak_order.remove(action)
                    else:
                        self.round_task[280] = [self.player_to_be_shoot]

                    # 猎人开枪后导致游戏结束,清空round_task
                    if len(self.get_in_game_player_id(role_ids=[2, 3, 4])) == 0 \
                            or len(self.get_in_game_player_id(role_ids=[0])) == 0 \
                            or len(self.get_in_game_player_id(role_ids=[1])) == 0:
                        self.round_task = {}

            self.print_current_task()

        # 添加后续发言的狼人自爆环节
        if task_id in [210, 230] and self.allow_suicide is True:
            wolf_player = self.get_in_game_wolf()
            self.round_task[270] = wolf_player[:]
            self.print_current_task(270)

        if task_id == 270:
            if action != -1:
                self.player_to_be_suicide = action
                self.attributes[self.player_to_be_suicide]['status'] = 'suicide'
                self.attributes[player_id]['is_revealed'] = True
                self.round_task = {}
                self.logger.info('狼人自爆: 玩家 %d' % (action + 1))

        if len(self.round_task) == 0:

            dead_player = [i for i in [self.player_to_be_exiled, self.player_to_be_shoot, self.player_to_be_suicide] if
                           i != -1]
            if len(dead_player) == 0:
                dead_player = [-1]
            self.out_game_history[self.time] = dead_player

            self.game_result = self.get_game_result()
            if self.game_result == -1:
                self.time += 1
                self.gen_task()
        return

    def get_final_reward(self):
        final_reward = []
        # 0 狼人全死 1 神全死 2 村民全死
        for player_id in range(9):
            role_name = self.attributes[player_id]['role']
            if role_name == 'werewolf':
                if self.game_result == 0:
                    reward = -2
                else:
                    reward = 2
            else:
                if self.game_result == 0:
                    reward = 1
                else:
                    reward = -1
            final_reward.append(reward)

        return final_reward

    def step(self, action=None):
        # print(self.round_task)

        if isinstance(action, str):
            pass
        elif isinstance(action, list):
            pass
        else:
            action = int(action)
            if action == self.player_num:
                action = 'speak'
            elif action == -1:
                pass
            # 空action 在 feature 端定义为 10
            elif action == 10:
                action = -1
            else:
                pass

        if self.game_result > -1:
            logging.error('game has ended, please reset a new game')
            return

        if self.time % 2 == 0:
            self.night(action)
        else:
            self.day(action)
        self.num_step += 1

        reward = 0.
        done = False

        self.random_vote = False
        if self.game_result == -1:
            player_id, state, legal_action = self.get_state_and_legal_action()
        else:
            player_id = -1
            state = None
            state = self.get_state(player_id, True)
            legal_action = []
            done = True

            final_reward = self.get_final_reward()
            game_result = [0] * 3
            game_result[self.game_result] = 1

        info = self.get_info(player_id)
        if done:
            info['final_reward'] = final_reward
            info['game_result'] = game_result

        return player_id, state, legal_action, reward, done, info

    def dump_log_with_json(self, logfile):
        with open(logfile, "w", encoding="utf-8") as f:
            json.dump(self.game_log, f, ensure_ascii=False)
            f.close()

    def speak_data2text(self, speak_data):
        text_all = []
        for data, data_map in zip(speak_data, self.map_data2text):
            text = data_map[data]
            text_all.append(text)

        speak = '-'.join(text_all)

        return speak

    def gen_random_speak(self):
        speak_data = []
        for word in self.map_data2text:
            data = random.randint(0, len(word.keys()) - 1)
            speak_data.append(data)
        speak = self.speak_data2text(speak_data)

        return speak

    def gen_random_speak_with_data(self):
        speak_data = []
        for word in self.map_data2text:
            data = random.randint(0, len(word.keys()) - 1)
            speak_data.append(data)
        speak = self.speak_data2text(speak_data)

        return speak, speak_data


if __name__ == '__main__':
    env = WereWolf9(debug=True)
    num_game = 1000000
    game_results = []

    for i in range(num_game):
        player_id, state, legal_action, reward, done, info = env.reset()
        while not done:
            action = np.random.choice(legal_action)
            if action == 'speak':
                action = env.gen_random_speak()
            else:
                assert action in legal_action
            player_id, state, legal_action, reward, done, info = env.step(action)

