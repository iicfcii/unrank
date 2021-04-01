import read
import capture
import server

def main():
    replay = server.read()

    if replay is None:
        print('No submitted replay')
    else:
        id = replay.get('objectId')
        code = replay.get('code')
        print('Replay id', id)
        print('Replay code', code)

        print('Recording', id)
        success = capture.record(code, id)

        if not success:
            print('Failed', id)
            server.fail(replay)
            return

        capture.screenshot(id)
        print('Processing', id)
        read.save_data(id)
        read.convert_csv(id)

        server.save(replay)
        print('Finished', id)
        
main()
