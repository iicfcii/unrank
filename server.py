import leancloud

# Status
STAT_SUBMITTED = 0
STAT_PROCESSING = 1
STAT_FINISHED = 2
STAT_FAILED = 3

leancloud.init('vNMcfOQkbIrK2mv1HCODUDie-MdYXbMMI', 'UDpxy5slHSvQQUzenUabBURS')

def read():
    replayQ = leancloud.Query('Replay')
    replayQ.equal_to('status', STAT_SUBMITTED)
    replayQ.ascending('createdAt')
    replay = replayQ.first()

    # replay.set('status',STAT_PROCESSING)
    # replay.save()

    return replay
