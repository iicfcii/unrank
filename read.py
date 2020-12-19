import assult
import control
import escort
import hybrid
import elim
import hero
import ult
import map
import utils
import matplotlib.pyplot as plt

code = 'numbani'

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
    if map_type == 'assult':
        assult.save(0,None,code)
    if map_type == 'control':
        control.save(0,None,code)
    if map_type == 'escort':
        escort.save(0,None,code)
    if map_type == 'hybrid':
        hybrid.save(0,None,code)
    obj_data = utils.load_data('obj',0,None,code)

obj_r_data = utils.load_data('obj_r',0,None,code)
if obj_r_data is None:
    if map_type == 'assult':
        assult.refine(code)
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
    hero_data = utils.load_data('hero_r',0,None,code)

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

assert elim_r_data['heroes'] == hero_r_data['heroes']

heroes_data = elim_r_data['heroes']
del elim_r_data['heroes']
del hero_r_data['heroes']

full_data = {
    'objective': obj_r_data,
    'heros': heroes_data,
    'hero': hero_r_data,
    'health': health_r_data,
    'ult': ult_r_data,
    'ult_use': ult_use_data,
    'elim': elim_r_data,
}

utils.save_data('full', full_data, 0, None, code)
