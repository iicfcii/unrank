import leancloud

import utils

# Status
STAT_SUBMITTED = 0
STAT_PROCESSING = 1
STAT_FINISHED = 2
STAT_FAILED = 3

leancloud.init('vNMcfOQkbIrK2mv1HCODUDie-MdYXbMMI', 'UDpxy5slHSvQQUzenUabBURS')

def replay(id):
    replayQ = leancloud.Query('Replay')
    replayQ.equal_to('objectId', id)
    replay = replayQ.first()

    return replay

def read(include_processing=False):
    replayQ = leancloud.Query('Replay')

    statuses = [STAT_SUBMITTED]
    if include_processing: statuses.append(STAT_PROCESSING)
    replayQ.contained_in('status', statuses)

    replayQ.ascending('createdAt')
    replays = replayQ.find()
    if len(replays) == 0: return None

    replay = replays[0]
    replay.set('status',STAT_PROCESSING)
    replay.save()

    return replay

def save(replay):
    id = replay.get('objectId')
    length = utils.count_frames(id)
    file_name = utils.file_path('full',0,(length-1),id,ext='json')
    with open(file_name, 'rb') as f:
        # Default application/json causes cors issues.
        file = leancloud.File(file_name.split('\\')[-1], f, 'application/octet-stream')
        file.save()

    csv_file_name = utils.file_path('full',0,(length-1),id,ext='csv')
    with open(csv_file_name, 'rb') as f:
        csv_file = leancloud.File(csv_file_name.split('\\')[-1], f, 'application/octet-stream')
        csv_file.save()

    replay.set('json', file)
    replay.set('csv', csv_file)
    replay.set('status', STAT_FINISHED)
    replay.save()

def fail(replay):
    replay.set('status', STAT_FAILED)
    replay.save()
