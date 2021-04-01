import read
import capture
import server

replay = server.read()
code = replay.get('code')
print('Replay code', code)

print('Recording', code)
# capture.record(code)
# capture.screenshot(code)
print('Processing', code)
read.save_data(code)
read.convert_csv(code)

# Upload and link file
