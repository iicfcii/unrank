import read
import capture
import server

replay = server.read()
code = replay.get('code')
print('Replay code', code)

print('Start recording', code)
# capture.record(code)
# capture.screenshot(code)
print('Processing data', code)
read.save_data(code)
read.convert_csv(code)

# Upload and link file
