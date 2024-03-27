# coding=utf-8
import copy
import glob
import os
import random
import shutil
import numpy as np
from tqdm import tqdm
import json
import csv
import re
import argparse
from collections import Counter
import jieba
import jieba.posseg
import matplotlib.pyplot as plt
# from env9 import WereWolf9
from env9_en import WereWolf9

import sys
sys.path.append('../')
from utils import (
    dict_pid, query, query_chatglm3, query_baichuan2,
    BAICHUAN_SERVER, CHATGLM_SERVER, GEN_SUMMARY_PROMPT,
    NON_RESULT_KEY, DICT_ID, jieba_word, class_dict,
    prob_dict, ROLE2FEATURE, ACTION2FEATURE, PROB2FEATURE,
    format2ner_v2
)


def random_select_audio(info_all):
    for name in info_all.keys():
        audios = info_all[name]['audio']
        audio_names = list(audios.keys())
        if len(audio_names) == 0:
            continue
        while True:
            random_audio = random.choice(audio_names)
            if audios[random_audio]['duration'] > 10 and audios[random_audio]['rms'] > 1e-4:
                print(name, random_audio)

                path_from = os.path.join(args.path_audio, name, random_audio + '.mp3')
                path_to = '../audio_test/' + name + '-' + random_audio + '.mp3'
                shutil.copy(path_from, path_to)
                break


def check_game_state(game_id, data):
    game_state = data['game_state']
    task_times = list(data['task_time'].keys())

    if 'Day 1 Night' not in game_state.keys():
        print('Incomplete replay information: ', game_id)
        return False

    is_complete = False
    for i in range(len(task_times)):
        task_time = task_times[i]
        if 'Day 1 Night' in task_time or 'Day 1 Daytime - Daybreak has come' in task_time:
            is_complete = True
            break
        elif i == 0 and 'Day 1 Daytime' in task_time and 'is preparing to speak' in task_time:
            # need to distinguish whether it's the start of the second round of speeches or not.
            if 'Voting Pattern (Round 2)' not in game_state['Day 1 Daytime'].keys():
                is_complete = True
            else:
                exile_id = game_state['Day 1 Daytime']['Voting Result']
                if 'Day 1 Daytime-[%d] is speaking(Round 2)' % exile_id in task_time:
                    is_complete = True
            break

    return is_complete


MAX_SPEAK_NUM = 0
MAX_SPEAK_NUM_PLAYER = 0

def check_audio(game_id, data):
    global MAX_SPEAK_NUM
    global MAX_SPEAK_NUM_PLAYER

    audios = data['audio']
    tasks = list(audios.keys())
    if len(tasks) > MAX_SPEAK_NUM:
        MAX_SPEAK_NUM = len(tasks)

    player_speak_num = [0] * 9
    for task in tasks:
        id = int(task.split('[')[1].split(']')[0]) - 1
        player_speak_num[id] += 1
    max_player_speak_num = max(player_speak_num)
    if max_player_speak_num > MAX_SPEAK_NUM_PLAYER:
        MAX_SPEAK_NUM_PLAYER = max_player_speak_num

    audio_key = 'text'
    first_none_id = -1
    for i in range(len(tasks)):
        task = tasks[i]
        audio = audios[task]
        if audio_key in audio.keys():
            text = audio['text']
            if len(text) > 0:
                first_none_id = i
                break
        # else:
        #     print(game_id, task, 'missing audio key', audio_key)


    # Check the first voice message of the complete game.
    # If the first voice message has sound, the whole game is likely to be normal.
    if first_none_id != 0:
         return False

    is_complete = True
    for i in range(len(tasks)):
        task = tasks[i]
        audio = audios[task]

        if audio_key not in audio.keys() or audio[audio_key] is None:
            is_complete = False
            print('speech %s is not processed: ' % audio_key, game_id, task, audio)
            break
        elif audio[audio_key] == '':
            task_split = task.split(']')[0] + ']'
            if task_split in data['game_state']['Automatically Passed for Not Speaking']:
                pass
                # print('Automatically passed the mic for not speaking ', game_id, audio_key, audio)
            else:
                if audio['duration'] > 5:
                    if abs(audio['duration'] - 15) < 1:
                        pass
                        # print('Suspected of automatically passing the mic for not speaking ', game_id, audio_key, audio)
                    else:
                        if first_none_id == 0:
                            pass
                            # print('Incomplete speech ', game_id, task, audio)
                        is_complete = False
                        break

    return is_complete


def check_label(game_id, data):
    audios = data['audio']
    tasks = list(audios.keys())

    is_complete = True
    for i in range(len(tasks)):
        task = tasks[i]
        audio = audios[task]

        if 'text' not in audio.keys():
            is_complete = False
            break

        text = audio['text']
        if len(text) == 0:
            continue

        if 'summary' not in audio.keys():
            is_complete = False
            break

        summary = audio['summary']
        if summary in ['', None]:
            is_complete = False
            print(audio)
            break

        if 'format' not in audio.keys():
            is_complete = False
            break

        format = audio['format']
        if format in ['', None]:
            is_complete = False
            print(audio)
            break

    return is_complete


def count_time(data_all):
    game_ids = list(data_all.keys())

    total_game_time = 0
    total_speak_time = 0
    total_speak_sample = 0
    for i in range(len(game_ids)):
        game_id = game_ids[i]
        data = data_all[game_id]

        if 'video' in data.keys():
            total_game_time += data['video']['time']
        # else:
        #     print(game_id, 'no video info')

        audios = data['audio']
        for audio in audios.values():
            total_speak_time += audio['duration']
            total_speak_sample += 1

    print('total game time:', total_game_time / 3600)
    print('total speech time:', total_speak_time / 3600)
    print('total speech sample:', total_speak_sample)


def check_complete(data_all):
    game_ids = list(data_all.keys())
    is_complete_num = [0, 0, 0, 0]
    for i in range(len(game_ids)):
        game_id = game_ids[i]
        data = data_all[game_id]


        # Night 1: Witch
        if 'Witch Used Antidote' not in data['game_state']['Day 1 Night']:
            pass
            # print('The witch does not save anyone on the first night: ', game_id)
        if 'Witch Used Poison' in data['game_state']['Day 1 Night']:
            pass
            # print('The witch uses her poison on the first night: ', game_id)

        is_complete_game = check_game_state(game_id, data)
        is_complete_audio = check_audio(game_id, data)
        is_complete_label = check_label(game_id, data)

        if is_complete_game:
            is_complete_num[0] += 1  # complete game
        if is_complete_audio:
            is_complete_num[1] += 1  # complete speech
        if is_complete_label:
            is_complete_num[2] += 1  # overall game

        if is_complete_game and is_complete_audio and is_complete_label:
            is_complete = True
            is_complete_num[-1] += 1
        else:
            is_complete = False

        data['is_complete'] = is_complete
        data['is_complete_game'] = is_complete_game
        data['is_complete_audio'] = is_complete_audio
        data['is_complete_label'] = is_complete_label

    print('num_full:', is_complete_num, 'num_game:', len(game_ids))
    print('Add complete information to data_all.')
    return


def load_data(update=True):
    path_data = '../data_all.json'
    if update or not os.path.exists(path_data):
        path_jsons = sorted(glob.glob(os.path.join(args.path_processed, '*.json')))
        data_all = {}
        for i in tqdm(range(len(path_jsons))):
            path_json = path_jsons[i]
            file_name = os.path.basename(path_json).split('.json')[0]
            data = load_json(path_json)
            data_all[file_name] = data

        print('save data to', path_data)
        save_json(path_data, data_all)
    else:
        data_all = load_json(path_data)

    return data_all


def sample_speaks(data_all, num=100):
    game_ids = list(data_all.keys())
    random.shuffle(game_ids)

    sample_speaks = []
    for game_id in game_ids:
        audios = data_all[game_id]['audio']
        tasks = list(audios.keys())
        random.shuffle(tasks)
        task = tasks[0]
        text = audios[task]['text']
        if text not in ['', None] and len(text) > 200:
            sample_speaks.append([game_id, task, text])
            if len(sample_speaks) >= num:
                break

    headers = ['game_id', 'task', 'speak', 'summary']

    # write sorted data to CSV file
    with open('../sample_speaks.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in sample_speaks:
            writer.writerow(row)


def save_json(path, data, encoding='utf-8-sig'):
    with open(path, 'w', encoding=encoding) as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return


def load_json(path, encoding='utf-8-sig'):
    with open(path, 'r', encoding=encoding) as f:
        data = json.load(f)
    return data

def quote_chinese(text):
    # match characters not in quotes
    pattern = r'([\u4e00-\u9fa5]+)'
    # use regular expressions to find matches and add quotes to them
    text = re.sub(pattern, r'"\1"', text)
    text = text.replace('\"\"', '\"')
    return text


def extract_number_from_brackets(s):
    numbers = re.findall(r'\[(\d+)\]', s)
    return int(numbers[0]) if numbers else None


def remove_before_speech(text: str) -> str:
    keyword = "："
    if keyword in text:
        return text.split(keyword, 1)[1]
    return text


def paraphrase_summary(sample,
                       baichuan=False,
                       chatglm=False,
                       chatgpt=True,
                       baichuan_server=BAICHUAN_SERVER,
                       chatglm_server=CHATGLM_SERVER):

    speaker = extract_number_from_brackets(sample['task'])

    prompt = (
        f"Player {dict_pid[speaker]} speaks: {sample['label']}\n\n"
        "You are now the Werewolf player number {dict_pid[speaker]}. Please directly paraphrase this statement:\n"
        "Requirements:\n"
        "1. If it includes third-person references to oneself, change them to first-person in the paraphrase. Keep the statement concise and avoid making things up.\n"
        "2. If there are no third-person references to oneself, you can directly output a summary of the original statement.\n"
        "Ensure the paraphrase is accurate, please directly output the paraphrasing result:"
    )

    if baichuan:
        response = query_baichuan2(prompt, baichuan_server)
    if chatglm:
        response = query_chatglm3(prompt, chatglm_server)
    if chatgpt:
        response = query(prompt)

    response = response.replace('“', '').replace('”', '')
    response = remove_before_speech(response)

    sample['label_raw'] = sample['label']
    sample['label'] = response

    return sample


for word in jieba_word + list(class_dict.keys()) + list(prob_dict.keys()):
    jieba.add_word(word)
    jieba.suggest_freq(word, True)


def format2ner(text_format):
    if text_format in [None, '']:
        return

    text_format = text_format.replace('?', '')
    text_format = text_format.replace('，', ',')
    text_format = text_format.replace(', ', ',')

    text_format = text_format.replace('PK', '特殊字符决斗')
    text_format = text_format.replace('pk', '特殊字符决斗')
    text_format = text_format.replace('None', '')
    text_format = text_format.replace('null', '')
    text_format = text_format.replace('NULL', '')
    text_format = text_format.replace('unknown', '')

    text_format = text_format.replace('[,]', '[]')
    text_format = text_format.replace('[,,]', '[]')
    text_format = text_format.replace('[,,,]', '[]')
    text_format = text_format.replace(',]', ']')
    text_format = text_format.replace('[,', '[')

    if '{' not in text_format:
        return
    text_format = text_format.split('{')[1]
    if '}' not in text_format:
        return
    text_format = text_format.split('}')[0]
    text_format = '{' + text_format + '}'

    text_format = quote_chinese(text_format)

    text_format = text_format.replace('特殊字符决斗', 'PK')
    text_format = text_format.replace('{"无结果"}', '{"无结果":[]}')
    text_format = text_format.replace(',"无结果"}', ',"无结果":[]}')

    try:
        data = json.loads(text_format)
    except Exception as e:
        # print('wrong format', text_format)
        return

    text_new = []
    data_new = {}
    for key in data.keys():
        value = data[key]
        if isinstance(value, list):
            pass
        elif isinstance(value, int):
            value = [value]
        else:
            try:
                value = [DICT_ID[value]]
            except:
                # print('wrong value', value)
                return

        ids = []
        for id in value:
            if isinstance(id, int):
                pass
            elif isinstance(id, list):
                # 暂时不处理 list in list
                # print('wrong list', id)
                return
            else:
                try:
                    id = DICT_ID[id]
                except:
                    # print('wrong id', id)
                    continue

            if id not in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                continue

            ids.append(id)

        if len(ids) > 0:
            data_new[key] = ids
            ids_str = key + '-' + ','.join(str(id) for id in ids)
            text_new.append(ids_str)

    if len(text_new) > 0:
        text_new = ' '.join(text_new)
    else:
        text_new = '无结果'

    return text_new, data_new


def _is_role(label):
    return label in ROLE2FEATURE.keys()


def _is_action(label):
    return label in ACTION2FEATURE.keys()


def _process_class_and_prob(key_class, key_prob):

    good_roles = {'预言家', '女巫', '猎人', '村民', '好人', '神职', '金水', '银水'}
    wolf_roles = {'狼人'}
    irrelevant_actions = {'查验', '救', '毒', '开枪', '刀口', '自爆', '投票', '弃票'}

    if '可能是' in key_class:
        key_class = key_class.replace('可能是', '')
        key_prob = '可能是'

    if key_prob in ['可能不是', '不是']:
        key_prob = key_prob.replace('不', '')
        if key_class in good_roles:
            key_class = '狼人'
        elif key_class in wolf_roles:
            key_class = '好人'
        elif key_class in irrelevant_actions:
            key_class = '无关信息'
        else:
            print(key_class)
            assert False

    elif key_prob == '不确定':
        key_class = '不确定身份'
        key_prob = '是'

    return key_class, key_prob


def _process_keys(key, id, feature_dict):
    data = load_json('key2data.json')

    if key not in data:
        data[key] = {'data': [['未处理', 'default']], 'num': 1}


    for key_class, key_prob in data[key]['data']:
        key_class, key_prob = _process_class_and_prob(key_class, key_prob)

        classes = set(ROLE2FEATURE.keys()) | set(ACTION2FEATURE.keys())

        if key_class not in classes:
            if key_class not in {'未处理', '无关信息', '弃票'}:
                print('feature missing', key_class)
            continue

        if id not in feature_dict:
            feature_dict[id] = {
                '身份': [[], []],
                '动作': [[], []],
            }

        if '可能' in key_prob:
            key_prob = key_prob.replace('可能', '')

        class_type = '身份' if _is_role(key_class) else '动作'

        feature_dict[id][class_type][0].append(key_class)
        feature_dict[id][class_type][1].append(key_prob)


    return feature_dict


def _extract_items_from_value(key, value):
    if isinstance(value, list) and value:
        if isinstance(value[0], list):
            try:
                return [[key, item[1]] for item in value]
            except TypeError as e:
                return [[key, value[0][1]]]
        else:
            return [[key, value[-1]]]
    return []


def _deduplicate(data):
    seen = set()
    return [item for item in data if tuple(item) not in seen and not seen.add(tuple(item))]


def gen_ner(data_all):
    game_ids = list(data_all.keys())
    for i in range(len(game_ids)):
        game_id = game_ids[i]
        data = data_all[game_id]
        audios = data['audio']
        for task in audios.keys():
            audio = audios[task]
            if 'format' not in audio.keys():
                continue
            text_format = audio['format']
            text_summary = audio['summary']

            if text_summary in ['', None]:
                continue
            if text_format in ['', None]:
                continue

            # return_format = format2ner(text_format)
            return_format = format2ner_v2(text_format)
            if return_format in [None]:
                continue

            ner_dict = return_format

            # audio['ner_text'] = ner_text
            audio['ner_dict'] = ner_dict

    return


def gen_data_format(data_all, max_len_content=150, max_len_label=150, summary2ner=True):
    rewrite_summary = False
    game_ids = list(data_all.keys())
    data_list = []


    for i in tqdm(range(len(game_ids))):
        game_id = game_ids[i]
        data = data_all[game_id]
        audios = data['audio']

        for task in audios.keys():
            player_id = int(task.split('[')[1].split(']')[0])
            header = 'Player %d:' % player_id
            audio = audios[task]
            if 'ner_text' not in audio.keys():
                continue

            content = audio['summary']
            label = audio['ner_text']

            if not summary2ner:
                content, label = label, content

            content = header + content
            if len(content) > max_len_content or len(label) > max_len_label:
                continue

            sample = {
                'game_id': game_id,
                'task': task,
                'content': content,
                'label': label,
            }
            if not summary2ner and rewrite_summary:
                sample = paraphrase_summary(sample, baichuan=True, chatglm=False)

            data_list.append(sample)


    print('Total number of games:', len(game_ids), 'Total number of samples:', len(data_list))
    random.shuffle(data_list)

    index_split = min(int(len(data_list) * 0.1), 10000)

    data_train = data_list[index_split:]
    data_test = data_list[:index_split]

    if summary2ner:
        file_tag = 'summary2ner'
    else:
        file_tag = 'ner2summary'

    save_json(file_tag + '_train.json', data_train, encoding='utf-8')
    save_json(file_tag + '_test.json', data_test, encoding='utf-8')

    return


def gen_data_format_new(data_all, max_len_content=150, max_len_label=150, summary2ner=True):
    rewrite_summary = False
    game_ids = list(data_all.keys())
    data_list = []

    for i in tqdm(range(len(game_ids))):
        game_id = game_ids[i]
        data = data_all[game_id]
        audios = data['audio']

        for task in audios.keys():
            player_id = int(task.split('[')[1].split(']')[0])
            speaker_id = dict_pid[player_id]
            header = 'Player %d:' % player_id

            audio = audios[task]
            if 'format_new' not in audio.keys():
                continue

            content = audio['summary']
            label = audio['format_new']

            task_prompt = (
                f"Now you are player number {speaker_id},"
                f"I am providing you with the key information for your upcoming speech:\n {label} \n"
                "It's your turn to speak now\n"
            )

            if not summary2ner:
                content, label = label, content

            try:
                content = GEN_SUMMARY_PROMPT + task_prompt
            except TypeError as e:
                print('Error in game_id:', game_id, 'content:', content, 'label:', label)


            sample = {
                'game_id': game_id,
                'task': task,
                'content': content,
                'label': label,
            }
            if not summary2ner and rewrite_summary:
                sample = paraphrase_summary(sample, baichuan=True, chatglm=False)

            data_list.append(sample)

    print('Total number of games:', len(game_ids), 'Total number of samples:', len(data_list))
    random.shuffle(data_list)

    index_split = min(int(len(data_list) * 0.1), 10000)

    data_train = data_list[index_split:]
    data_test = data_list[:index_split]

    if summary2ner:
        file_tag = 'summary2ner'
    else:
        file_tag = 'ner2summary'

    save_json(file_tag + '_train.json', data_train, encoding='utf-8')
    save_json(file_tag + '_test.json', data_test, encoding='utf-8')

    return


def gen_data_llm(data_all, max_len_content=150, max_len_label=150):
    print('-' * 100)
    print('NER generated')
    # gen_data_format(data_all, max_len_content, max_len_label, summary2ner=True)
    # gen_data_format(data_all, max_len_content, max_len_label, summary2ner=False)
    gen_data_format_new(data_all, max_len_content, max_len_label, summary2ner=True)
    gen_data_format_new(data_all, max_len_content, max_len_label, summary2ner=False)


def gen_key_map(data_all, plot_hist=False, print_key=False):

    print('-' * 100)
    print('Tokenization')

    game_ids = list(data_all.keys())

    all_keys = []
    cut_keys = []
    cut_keys_flatten = []
    for i in range(len(game_ids)):
        game_id = game_ids[i]
        data = data_all[game_id]
        audios = data['audio']
        for task in audios.keys():
            audio = audios[task]
            if 'ner_dict' not in audio.keys():
                continue
            data = audio['ner_dict']
            try:
                for item in ['身份类型', '动作类型']:
                    for key in data[item].keys():
                        all_keys.append(key)
                        key_cut = jieba.posseg.lcut(key)
                        tmp = []
                        for word, posseg in key_cut:
                            tmp.append(word)
                        cut_keys.append(tmp)
                        cut_keys_flatten += tmp
            except KeyError as e:
                continue

    count_key = Counter(all_keys)
    count_cut = Counter(cut_keys_flatten)
    print('Number of key categories before tokenization:', len(count_key))
    print('Number of key categories after tokenization:', len(count_cut))

    if plot_hist:
        plt.figure(1)
        frequency = np.array(list(count_cut.values()))
        frequency = np.log10(frequency)
        plt.hist(frequency, bins=100)
        plt.yscale("log")
        plt.xlabel('key frequency (log10)')
        plt.ylabel('key types number')
        plt.show()

    class_keys = list(class_dict.keys())
    prob_keys = list(prob_dict)

    filter_keys = {}
    key2data = {}
    count_done = 0
    for tmp, all_key in zip(cut_keys, all_keys):
        if all_key in key2data.keys():
            key2data[all_key]['num'] += 1

        cut_key = tmp[:]  # copy for later removal

        # first, check class
        results_class = []
        for key in cut_key:
            for class_key in class_keys:
                if class_key == key:
                    results_class.append(class_key)
                    # if key has already been classified into class words, do not count it again for prob words
                    # cut_key.remove(key)
                    break
        if len(results_class) == 0:
            results_class.append('default')

        # then, check prob
        results_prob = []

        for key in cut_key:
            for prob_key in prob_keys:
                if prob_key == key:
                    results_prob.append(prob_key)
                    break
        if len(results_prob) == 0:
            results_prob.append('default')

        if results_class != ['default']:
            count_done += 1

        if len(results_prob) > 1:
            print(cut_key, results_prob)

        data_list = []
        for result_class in results_class:
            for result_prob in results_prob:
                class_key_values = class_dict[result_class]
                prob_key_value = prob_dict[result_prob]

                if isinstance(class_key_values, str):
                    class_key_values = [class_key_values]
                for class_key_value in class_key_values:
                    if class_key_value in ['未处理', '无关信息']:
                        prob_key_value = 'default'

                    if class_key_value not in filter_keys.keys():
                        filter_keys[class_key_value] = {}
                    if prob_key_value not in filter_keys[class_key_value].keys():
                        filter_keys[class_key_value][prob_key_value] = []

                    filter_keys[class_key_value][prob_key_value].append(all_key)
                    data_list.append([class_key_value, prob_key_value])

        key2data[all_key] = {
            'num': 1,
            'data': data_list,
        }


    print('Processed keys:', count_done, 'Total number of keys:', len(all_keys),
          'Percentage: %.2f' % (100 * count_done / len(all_keys)))

    class_types = []
    for value in class_dict.values():
        if isinstance(value, list):
            class_types += value
        else:
            class_types.append(value)
    class_types = list(set(class_types))
    prob_types = list(set(list(prob_dict.values())))

    print('number of class categories:', len(class_types))
    print('class:', class_types)
    print('number of prob categories:', len(prob_types))
    print('prob:', prob_types)

    if print_key:
        # sort the internal dict first
        sorted_inner_data = {
            key: {k: v for k, v in sorted(value.items(), key=lambda x: len(x[1]), reverse=True)}
            for key, value in filter_keys.items()
        }

        # calculate the sum of the lengths of the values of each internal dict
        length_sums = {
            key: sum(len(value) for value in inner_dict.values())
            for key, inner_dict in sorted_inner_data.items()
        }

        length_sums['无关信息'] = -1
        length_sums['未处理'] = -2

        # Sort external dict by sum of lengths
        filter_keys_sort = {
            key: sorted_inner_data[key]
            for key in sorted(length_sums, key=lambda x: length_sums[x], reverse=True)
        }

        for class_key in filter_keys_sort.keys():
            for prob_key in filter_keys_sort[class_key].keys():
                key_list = filter_keys_sort[class_key][prob_key]
                if len(key_list) == 0:
                    continue
                count = Counter(key_list)
                print('-' * 100)
                print('class:', class_key, 'prob:', prob_key)
                print(f'num: {len(key_list)}')


    key2data_sort = {
        key: key2data[key] for key in sorted(key2data.keys(), key=lambda x: key2data[x]['num'], reverse=True)
    }

    save_json('key2data.json', key2data_sort)

    return key2data


ROLE_MAP = {
    'Villager': 'villager',
    'Werewolf': 'werewolf',
    'Seer': 'seer',
    'Witch': 'witch',
    'Hunter': 'hunter',
}


def get_speak_seq(data):

    # get the speaking order
    audios = data['audio']
    game_state = data['game_state']
    roles = game_state['roles']
    task_speaks = list(audios.keys())
    speak_seq = {}

    # it needs to be pre-initialized because there might be a day but nothing is said,
    # like the hunter shoots and the game is over
    for key in game_state.keys():
        if 'Daytime' in key:
            speak_seq[key] = {
                "Hunter shoot at night": -1,
                "First round of speeches": [],
                "Second round of speeches": [],
                "Last words for been exiled": [],
                "Last words for been shot": [],
                "Last words for died at night": [],
            }
    for task_speak in task_speaks:
        time, id = task_speak.split('-')
        # time = TIME_MAP[time.strip()]
        time = time.strip()
        id = int(id.split(']')[0].split('[')[1])
        assert id in [1, 2, 3, 4, 5, 6, 7, 8, 9]

        if 'is speaking (round 2)' in task_speak:
            speak_seq[time]["Second round of speeches"].append(id)
        elif 'is speaking' in task_speak:
            speak_seq[time]["First round of speeches"].append(id)
        elif 'gives the last words' in task_speak:
            if 'Activate ability' in game_state[time].keys() and game_state[time]['Activate ability'] == id:
                speak_seq[time]["Last words for been shot"].append(id)
            elif id in game_state[time.replace('Daytime', 'Night')]['Death Message']:
                speak_seq[time]["Last words for died at night"].append(id)
            elif game_state[time]['Voting Result'] == id:
                speak_seq[time]["Last words for been exiled"].append(id)
            else:
                assert False
        else:
            assert False

        if 'Activate ability' in game_state[time].keys():
            if '公投结果' in game_state[time].keys():
                exile_id = game_state[time]['Voting Result']
                if exile_id != -1 and roles[str(exile_id)] != '猎人':
                    speak_seq[time]["Hunter shoot at night"] = game_state[time]['Activate ability']

    return speak_seq


def get_action(game_id, data, player_id, state, legal_action, speak_seq):
    game_state = data['game_state']
    audios = data['audio']

    task_name = state["game"]["task_name"]
    time = state["game"]["time"]
    time_str = 'Day ' + str(time // 2 + 1)
    if time % 2 == 0:
        time_str += ' Night'
    else:
        time_str += ' Daytime'


    player_id_data = player_id + 1

    if task_name == 'Werewolf vote for kill':
        action = game_state[time_str]['Werewolf']
    elif task_name in ['Witch antidote', 'Witch poison']:
        if task_name in game_state[time_str].keys():
            action = game_state[time_str][task_name]
        else:
            action = -1
    elif task_name == 'Seer check':
        action = game_state[time_str]['Seer']
    elif task_name == 'Werewolf commit suicide':
        if 'suicide' not in game_state[time_str].keys():
            action = -1
        else:
            suicide_id = game_state[time_str]['suicide']
            if player_id_data == suicide_id:
                if len(speak_seq[time_str]["First round of speeches"]) == 0 and len(speak_seq[time_str]["Second round of speeches"]) == 0:
                    action = suicide_id
                else:
                    action = -1
            else:
                action = -1

    elif task_name in ["First round of speeches", "Second round of speeches", "Last words for been exiled", "Last words for been shot", "Last words for died at night"]:
        if task_name == "First round of speeches":
            speak_type = 'is speaking'
        elif task_name == "Second round of speeches":
            speak_type = 'is speaking (round 2)'
        else:
            speak_type = 'gives the last words'

        task_speak_env = time_str + ' - [%d] %s' % (player_id_data, speak_type)
        # task_speak_ori = task_speak_env.replace('日', '天白天').replace('夜', '天黑夜')
        task_speak_ori = task_speak_env

        if task_speak_ori not in audios.keys():
            action = ''
        else:
            if 'feature' not in audios[task_speak_ori].keys():
                action = audios[task_speak_ori]['text']
            else:
                action = audios[task_speak_ori]['feature'].tolist()
            speak_seq[time_str][task_name].remove(player_id_data)
    elif task_name == 'First round of vote':
        action = game_state[time_str]['Voting Pattern'][str(player_id_data)]
    elif task_name == 'Second round of vote':
        action = game_state[time_str]['Voting Pattern (Round 2)'][str(player_id_data)]
    elif task_name == 'Hunter shoot at night':
        if 'Activate ability' not in game_state[time_str].keys():
            action = -1
        else:
            action = game_state[time_str]['Activate ability']
    else:
        assert False

    if isinstance(action, int):
        if action != -1:
            action = action - 1
        if action not in legal_action:
            print(action, legal_action)
            assert False
    return action


def simulate_game(env, game_id, data):

    print(f'game id: {game_id}')

    game_state = data['game_state']
    roles = game_state['roles']

    role_list = [None] * 9
    for id_str in roles.keys():
        id = int(id_str) - 1
        role = roles[id_str]
        role_eng = ROLE_MAP[role]
        role_list[id] = role_eng

    final_status = game_state['final']
    speak_seq = get_speak_seq(data)
    data['speak_seq'] = copy.deepcopy(speak_seq)

    player_id, state, legal_action, reward, done, info = env.reset(role_list=role_list, speak_seq=speak_seq, allow_suicide=True)
    while not done:
        action = get_action(game_id, data, player_id, state, legal_action, speak_seq)
        player_id, state, legal_action, reward, done, info = env.step(action)


    # Game over, check results
    final_status_env = state['status']
    for id_str in final_status.keys():
        id = int(id_str) - 1
        status = final_status[id_str]
        status_env = final_status_env[id]
        if status_env == 'saved':
            status_env = 'in_game'
        # status_env = env.status_eng[status_env]
        if status != status_env:
            print(id, 'fanlang:', status, 'env:', status_env)
            assert False

    return


def simulate_game_all(data_all):
    env = WereWolf9(debug=True)
    game_ids = list(data_all.keys())

    for i in range(0, len(game_ids)):
        game_id = game_ids[i]
        # game_id = '20230929130130_1688'

        data = data_all[game_id]
        is_complete = data['is_complete_game']
        if not is_complete:
            continue
        simulate_game(env, game_id, data)

    # game_id = '00f0717bc4157d80f86e4e20'
    # data = data_all[game_id]
    # simulate_game(env, game_id, data)


def rms_hist(data_all):
    game_ids = list(data_all.keys())

    rms_list = []
    for i in range(len(game_ids)):
        game_id = game_ids[i]
        data = data_all[game_id]
        audios = data['audio']
        for task in audios.keys():
            rms = max(audios[task]['rms'], 1e-6)
            rms_list.append(rms)

    print('num game', len(game_ids), 'num audio', len(rms_list), 'rms_max',min(rms_list), 'rms_mean', np.mean(rms_list))

    rms_log = np.log10(rms_list)

    plt.figure(1)
    plt.hist(rms_log, bins=100)
    plt.show()


def reverse_dict(d):
    new_dict = {}
    for key, value in d.items():
        if value not in new_dict:
            new_dict[value] = key
        else:
            new_dict[value] += '/' + key
    return new_dict

# FEATURE2CLASS = reverse_dict(CLASS2FEATURE)
FEATURE2ROLE = reverse_dict(ROLE2FEATURE)
FEATURE2ACTION = reverse_dict(ACTION2FEATURE)




def merge_multi_label(feature_dict):

    role_priority = ['狼人', '预言家', '女巫', '金水', '银水', '猎人', '神职', '好人', '不确定身份']
    action_priority = ['投票', '查验', '救', '刀口', '开枪', '毒', '自爆']

    for id in feature_dict.keys():
        roles, role_probs = feature_dict[id]['身份']
        if len(roles) > 1:
            for priority in role_priority:
                if priority in roles:
                    role_probs = [role_probs[roles.index(priority)]]
                    roles = [priority]
                    feature_dict[id]['身份'] = [roles, role_probs]
                    break

        actions, action_probs = feature_dict[id]['动作']
        if len(actions) > 1:
            for priority in action_priority:
                if priority in actions:
                    action_probs = [action_probs[actions.index(priority)]]
                    actions = [priority]
                    feature_dict[id]['动作'] = [actions, action_probs]
                    break

    return feature_dict


def dict2feature(feature_dict):
    feature_multi = np.zeros(shape=[9, len(ROLE2FEATURE) + len(ACTION2FEATURE)], dtype=np.int8)
    feature_single = np.zeros(shape=[9, 2], dtype=np.int8) - 1

    for id in feature_dict.keys():
        roles, role_probs = feature_dict[id]['身份']
        for role, role_prob in zip(roles, role_probs):
            feature_multi[id, ROLE2FEATURE[role]] = PROB2FEATURE[role_prob]
            feature_single[id, 0] = ROLE2FEATURE[role]

        actions, action_probs = feature_dict[id]['动作']
        for action, action_prob in zip(actions, action_probs):
            feature_multi[id, len(ROLE2FEATURE) + ACTION2FEATURE[action]] = PROB2FEATURE[action_prob]
            feature_single[id, 1] = ACTION2FEATURE[action]

    feature_single += 1

    return feature_multi, feature_single


def data2feature(audio_data, key2data_sort, merge):

    roles = list(ROLE2FEATURE.keys())
    actions = list(ACTION2FEATURE.keys())
    classes = roles + actions

    feature_dict = {}
    for key in audio_data.keys():
        tmps = key2data_sort[key]['data']
        ids = audio_data[key]
        for tmp in tmps:
            key_class, key_prob = tmp

            # 处理两个遗留问题 可能是好人  可能是狼人
            if '可能是' in key_class:
                key_class = key_class.replace('可能是', '')
                key_prob = '可能是'

            # 求解可能不是
            if key_prob in ['可能不是', '不是']:
                key_prob = key_prob.replace('不', '')
                if key_class in ['预言家', '女巫', '猎人', '村民', '好人', '神职', '金水', '银水']:
                    key_class = '狼人'
                elif key_class in ['狼人']:
                    key_class = '好人'
                elif key_class in ['查验', '救', '毒', '开枪', '刀口', '自爆', '投票', '弃票']:
                    key_class = '无关信息'
                else:
                    print(key_class)
                    assert False

            elif key_prob == '不确定':
                key_class = '不确定身份'
            if key_class == '不确定身份':
                key_prob = '是'
            if key_class not in classes:
                if key_class not in ['未处理', '无关信息', '弃票']:
                    print('feature 遗漏', key_class)
                continue

            for id in ids:
                id = int(id) - 1
                assert id in [0, 1, 2, 3, 4, 5, 6, 7, 8]

                if id not in feature_dict.keys():
                    feature_dict[id] = {
                        '身份': [[], []],
                        '动作': [[], []],
                    }

                if key_class in roles:
                    class_type = '身份'
                elif key_class in actions:
                    class_type = '动作'
                else:
                    assert False
                feature_dict[id][class_type][0].append(key_class)
                feature_dict[id][class_type][1].append(key_prob)

    if merge:
        # 合并多重动作类型和身份类型
        feature_dict = merge_multi_label(feature_dict)

    feature_multi, feature_single = dict2feature(feature_dict)

    return feature_multi, feature_single


def data2feature_v2(audio_data, key2data_sort, merge):
    result = []
    items = []

    identities = audio_data.get('身份类型', {})
    actions = audio_data.get('动作类型', {})

    for key, values in identities.items():
        if isinstance(values, list):
            for value in values:
                items.append([key, value if not isinstance(value, list) else value[0]])
        else:
            items.append([key, values])

    for key, values in actions.items():
        items.extend(_extract_items_from_value(key, values))

    # items = _deduplicate(items)

    feature_dict = {}
    for item in items:
        key, id = item
        if id not in ROLE2FEATURE.values():
            continue
        id = int(id) - 1
        feature_dict = _process_keys(key, id, feature_dict)

    if merge:
        feature_dict = merge_multi_label(feature_dict)

    try:
        feature_multi, feature_single = dict2feature(feature_dict)
        with open('record_feature_dict.jsonl', 'a', encoding='utf-8') as out:
            out.write(json.dumps(feature_dict, ensure_ascii=False) + '\n')
    except IndexError as e:
        print(f'feature_dict: {feature_dict}')
        raise e

    return feature_multi, feature_single


def gen_feature(data_all, merge=True):

    print('-'*100)
    print('generating feature...')
    print('feature map: ', FEATURE2ROLE, FEATURE2ACTION)

    game_ids = list(data_all.keys())
    key2data_sort = load_json('key2data.json')

    feature_all = []

    for i in range(len(game_ids)):
        game_id = game_ids[i]
        data = data_all[game_id]
        audios = data['audio']
        for task in audios.keys():
            audio = audios[task]
            if 'ner_dict' not in audio.keys():
                continue

            audio_data = audio['ner_dict']
            # feature_multi, feature_single = data2feature(audio_data, key2data_sort, merge)
            feature_multi, feature_single = data2feature_v2(audio_data, key2data_sort, merge)

            audio['feature'] = feature_single.flatten()
            feature_all.append(feature_multi)

    feature_all = np.concatenate(feature_all, axis=0)

    np.save('feature_all.npy', feature_all)

    feature_all_clip = np.clip(feature_all, 0, 1)
    print(np.max(np.sum(feature_all_clip, axis=-1)))

    data_feature = {}
    game_ids = list(data_all.keys())
    for i in range(len(game_ids)):
        game_id = game_ids[i]
        if data_all[game_id]['is_complete']:
            data_feature[game_id] = data_all[game_id]

    print('len data_feature', len(data_feature))

    return data_feature


def parse_args():
    parser = argparse.ArgumentParser(description='Process game data.')
    parser.add_argument('--path_json', default='../data/demo/game_data/', help='Path to game data JSON files.')
    parser.add_argument('--path_audio', default='../data/demo/audio/', help='Path to audio files.')
    parser.add_argument('--path_video', default='../data/demo/video/', help='Path to video files.')
    parser.add_argument('--path_processed', default='../data/demo/opensource/', help='Path to processed data.')
    parser.add_argument('--path_temp', default='../data/temp/', help='Path to temporary files.')
    parser.add_argument('--audio_key', default=None, choices=[None, 'text', 'summary', 'format'], help='Audio key for processing.')
    return parser.parse_args()

def main(args):
    random.seed(0)
    np.random.seed(0)

    # data_all = load_data(update=False)
    data_all = load_data(update=True)
    count_time(data_all)
    check_complete(data_all)

    gen_ner(data_all)
    # gen_key_map(data_all, print_key=True, plot_hist=True)

    # rms_hist(data_all)
    # sample_speaks(data_all)

    # gen_data_llm(data_all)
    # data_feature = gen_feature(data_all, merge=False)
    simulate_game_all(data_all)

    # path_data_feature = 'data_feature.npy'
    # np.save(path_data_feature, data_feature)
    # save_json(path_data_feature, data_feature)

    # print('max speak num', MAX_SPEAK_NUM)
    # print('max player speak num', MAX_SPEAK_NUM_PLAYER)

    # save_json('data_all_with_ner.json', data_all)


if __name__ == '__main__':
    args = parse_args()
    main(args)
