import read
import capture

code = '72M66E'
capture.record(code)
capture.screenshot(code)
read.save_data(code)
read.convert_csv(code)
