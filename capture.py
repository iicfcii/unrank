import time
import os
import subprocess
import cv2
import pyautogui as gui
import pyperclip

MENU_CAREER_PROFILE = (960,555)
MENU_LEAVE_GAME = (960,660)
WAIT_AFTER_CLICK = 1

# Start or focus the overwatch launcher(battlenet)
# subprocess.call(['E:\Overwatch\Overwatch Launcher.exe','--productcode=pro'])

def save_templates():
    # img = gui.screenshot(
    #     './template_ui/play_button.png',
    #     region=(400,720,210,40)
    # )

    # img = gui.screenshot(
    #     './template_ui/overwatch_logo.png',
    #     region=(46,54,450,50)
    # )

    # img = gui.screenshot(
    #     './template_ui/replay_button.png',
    #     region=(670,40,100,40)
    # )

    # img = gui.screenshot(
    #     './template_ui/import_button.png',
    #     region=(1680,330,110,30)
    # )

    # img = gui.screenshot(
    #     './template_ui/view_button.png',
    #     region=(960,600,110,40)
    # )

    # img = gui.screenshot(
    #     './template_ui/replay_item.png',
    #     region=(1830,420,35,30)
    # )

    # img = gui.screenshot(
    #     './template_ui/share_button.png',
    #     region=(1100,875,140,40)
    # )

    # img = gui.screenshot(
    #     './template_ui/copy_button.png',
    #     region=(960,615,135,45)
    # )

    # img = gui.screenshot(
    #     './template_ui/replay_exists.png',
    #     region=(660,480,600,50)
    # )

    # img = gui.screenshot(
    #     './template_ui/ok_button.png',
    #     region=(900,570,120,50)
    # )

    # img = gui.screenshot(
    #     './template_ui/small_view_button.png',
    #     region=(900,875,140,40)
    # )

    # img = gui.screenshot(
    #     './template_ui/replay_ends.png',
    #     region=(1500,940,300,100)
    # )
    pass

save_templates()

def read_tempaltes():
    templates = {}

    templates['play_button'] = './template_ui/play_button.png'
    templates['overwatch_logo'] = './template_ui/overwatch_logo.png'
    templates['replay_button'] = './template_ui/replay_button.png'
    templates['import_button'] = './template_ui/import_button.png'
    templates['view_button'] = './template_ui/view_button.png'
    templates['small_view_button'] = './template_ui/small_view_button.png'
    templates['replay_item'] = './template_ui/replay_item.png'
    templates['share_button'] = './template_ui/share_button.png'
    templates['copy_button'] = './template_ui/copy_button.png'
    templates['ok_button'] = './template_ui/ok_button.png'

    templates['replay_exists'] = './template_ui/replay_exists.png'
    templates['replay_ends'] = './template_ui/replay_ends.png'

    return templates

templates = read_tempaltes()

def wait_for(img, confidence=0.9):
    wait_time = 1
    location = None
    count = 0
    while location is None and count < 10/wait_time:
        location = gui.locateOnScreen(templates[img],confidence=confidence)
        time.sleep(wait_time)
        count += 1
    assert location is not None, 'Wait for {} image timeouts'.format(img)
    return location

def click(loc):
    gui.moveTo(loc)
    time.sleep(0.5)
    gui.click()

def find(code):
    location = None
    # Make sure in replay section
    for loc in gui.locateAllOnScreen(templates['replay_item']):
        x, y = gui.center(loc)
        click((x-50, y))
        loc = wait_for('share_button',confidence=0.8)
        click(gui.center(loc))
        loc = wait_for('copy_button',confidence=0.8)
        click(gui.center(loc))
        c = pyperclip.paste()
        if code == c:
            location = loc
            break
    return location

def record(code, id):
    try:
        # Import replay and start
        loc = wait_for('overwatch_logo',confidence=0.8) # Wait for game home screen
        click(gui.center(loc)) # Make sure focus the window
        gui.press('esc') # Enter option menu
        time.sleep(WAIT_AFTER_CLICK)
        click(MENU_CAREER_PROFILE) # Click career profile
        loc = wait_for('replay_button')
        click(gui.center(loc))
        loc = wait_for('import_button')
        click(gui.center(loc))
        time.sleep(WAIT_AFTER_CLICK) # Wait for menu to pop out
        gui.write(code, interval=0.25)
        time.sleep(WAIT_AFTER_CLICK)
        gui.press('enter') # Confirm input
        time.sleep(WAIT_AFTER_CLICK)

        loaded = 0
        loc = gui.locateOnScreen(templates['replay_exists'])
        if loc is None:
            loaded = 1 # New replay
        else:
            loc = wait_for('ok_button')
            click(gui.center(loc))

        if loaded == 0:
            loc = find(code)
            if loc is not None: loaded = 2 # Existing replay

        if loaded == 0: print('Failed to find the existing replay', code)
        assert loaded > 0, 'Replay can not be loaded'

        if loaded == 1:
            loc = wait_for('view_button')
            click(gui.center(loc))
        else:
            loc = wait_for('small_view_button')
            click(gui.center(loc))

        # Start screen recording
        folder_path = './vid'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        proc = subprocess.Popen(
            'ffmpeg -y -f gdigrab -r 30 -i title=Overwatch -c:v h264_nvenc -b:v 10M ./vid/{}.mp4'.format(id),
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Wait for several continuous replay ends screen to finish
        count = 0
        while count < 3:
            try:
                loc = wait_for('replay_ends')
                time.sleep(1)
                count += 1
            except:
                count = 0 # Reset counter if no replay ends seen

        # End screen recording
        proc.communicate(input='q'.encode())

        # Back to main screen
        gui.press('esc')
        time.sleep(WAIT_AFTER_CLICK)
        click(MENU_LEAVE_GAME)
        time.sleep(WAIT_AFTER_CLICK)
        gui.press('esc')
        loc = wait_for('overwatch_logo',confidence=0.8)

        return True
    except AssertionError as e:
        print(e)
        return False

def screenshot(code):
    folder_path = './img/{}'.format(code)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    proc = subprocess.Popen(
        'ffmpeg -i ./vid/{}.mp4 -vf fps=1,scale=-1:720 -start_number 0 ./img/{}/{}_%d.png'.format(code,code,code),
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    proc.wait()

# NOTE: A real mouse click is required to focus the keyboard
