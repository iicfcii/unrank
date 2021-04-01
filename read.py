import csv
import cv2
import os

import assault
import control
import escort
import hybrid
import elim
import hero
import ult
import map
import utils
import matplotlib.pyplot as plt

def save_data(code):
    folder_path = './data/{}'.format(code)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    map_name = None
    for img, frame in utils.read_frames(0,10,code):
        if map_name is None:
            map_name = map.read_map(img, map.read_templates())
        else:
            break
    assert map_name is not None
    map_type = map.MAPS[map_name]

    print(code, map_name, map_type)

    # Objective
    obj_data = utils.load_data('obj',0,None,code)
    if obj_data is None:
        if map_type == 'assault':
            assault.save(0,None,code)
        if map_type == 'control':
            control.save(0,None,code)
        if map_type == 'escort':
            escort.save(0,None,code)
        if map_type == 'hybrid':
            hybrid.save(0,None,code)
        obj_data = utils.load_data('obj',0,None,code)

    obj_r_data = utils.load_data('obj_r',0,None,code)
    if obj_r_data is None:
        if map_type == 'assault':
            assault.refine(code)
        if map_type == 'control':
            control.refine(code)
        if map_type == 'escort':
            escort.refine(code)
        if map_type == 'hybrid':
            hybrid.refine(code)
        obj_r_data = utils.load_data('obj_r',0,None,code)

    # Hero
    hero_data = utils.load_data('hero',0,None,code)
    if hero_data is None:
        hero.save(0, None, code)
        hero_data = utils.load_data('hero',0,None,code)

    hero_r_data = utils.load_data('hero_r',0,None,code)
    if hero_r_data is None:
        hero.refine(code)
        hero_r_data = utils.load_data('hero_r',0,None,code)

    # Ult
    ult_data = utils.load_data('ult',0,None,code)
    if ult_data is None:
        ult.save(0, None, code)
        ult_data = utils.load_data('ult',0,None,code)

    ult_r_data = utils.load_data('ult_r',0,None,code)
    if ult_r_data is None:
        ult.refine(code)
        ult_r_data = utils.load_data('ult_r',0,None,code)

    ult_use_data = utils.load_data('ult_use',0,None,code)
    if ult_use_data is None:
        ult.use(code)
        ult_use_data = utils.load_data('ult_use',0,None,code)

    # Elim
    elim_data = utils.load_data('elim',0,None,code)
    health_data = utils.load_data('health',0,None,code)
    if elim_data is None or health_data is None:
        elim.save(0, None, code)
        elim_data = utils.load_data('elim',0,None,code)

    elim_r_data = utils.load_data('elim_r',0,None,code)
    health_r_data = utils.load_data('health_r',0,None,code)
    if health_data is None or elim_r_data is None:
        elim.refine(code)
        elim_r_data = utils.load_data('elim_r',0,None,code)

    # assert elim_r_data['heroes'] == hero_r_data['heroes']

    heroes_data = elim_r_data['heroes']
    del elim_r_data['heroes']
    del hero_r_data['heroes']

    time_data = {
        'interval': 1,
        'data': [i for i in range(0,len(obj_r_data['status']))]
    }

    # TODO: Remove data until map is recognized
    full_data = {
        'map': map_name,
        'time': time_data,
        'objective': obj_r_data,
        'heroes': heroes_data,
        'hero': hero_r_data,
        'health': health_r_data,
        'ult': ult_r_data,
        'ult_use': ult_use_data,
        'elim': elim_r_data,
    }

    utils.save_data('full', full_data, 0, None, code)

def convert_csv(code):
    length = utils.count_frames(code)
    full_data = utils.load_data('full',0,None,code)
    file_name = utils.file_path('full', 0, (length-1)*30, code, ext='csv')

    titles = []

    map_keys = ['map']
    titles += map_keys

    time_keys = ['interval', 'time']
    titles += time_keys

    objective_keys = list(full_data['objective'].keys())
    if full_data['objective']['type'] != 'escort':
        objective_keys.remove('progress')
        objective_progress_keys = list(full_data['objective']['progress'].keys())
        objective_progress_keys = ['progress_{}'.format(i) for i in objective_progress_keys]
        objective_keys += objective_progress_keys
    titles += objective_keys

    heroes_keys = ['heroes']
    titles += heroes_keys

    for type in ['hero', 'health', 'ult', 'ult_use', 'elim']:
        keys = list(full_data[type].keys())
        keys = ['{}_{}'.format(type, i) for i in keys]
        titles += keys

    with open(file_name, mode='w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(titles)

        for i in range(length):
            row = []

            map_values = []
            map_values.append(full_data['map'] if i == 0 else None)
            row += map_values

            time_values = []
            for key in full_data['time']:
                if key == 'interval':
                    time_values.append(full_data['time']['interval'] if i == 0 else None)
                else:
                    time_values.append(full_data['time'][key][i])
            row += time_values

            objective_values = []
            for key in full_data['objective']:
                if key == 'progress':
                    if full_data['objective']['type'] != 'escort':
                        for type in full_data['objective'][key]:
                            objective_values.append(full_data['objective'][key][type][i])
                    else:
                        objective_values.append(full_data['objective'][key][i])
                elif key == 'type':
                    objective_values.append(full_data['objective'][key] if i == 0 else None)
                else:
                    objective_values.append(full_data['objective'][key][i])
            row += objective_values

            heroes_values = []
            heroes_values.append(full_data['heroes'][i] if i < len(full_data['heroes']) else None)
            row += heroes_values

            for type in ['hero', 'health', 'ult', 'ult_use', 'elim']:
                values = []
                for key in full_data[type]:
                    values.append(full_data[type][key][i])
                row += values

            writer.writerow(row)

# code = 'Y1DZV3'
# save_data(code)
# convert_csv(code)
# Check error
# for src, frame in utils.read_frames(390, None, code):
#     info, img = hero.process_heroes(src)
#     print(info)
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
# Redo refine
# hero.refine(code)
