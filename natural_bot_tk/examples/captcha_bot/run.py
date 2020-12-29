from time import sleep
from io import BytesIO
import json

import pyautogui
import cv2
import numpy as np

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from natural_bot_tk import Mouse
from telegram_client import BotClient

def check_for_captchas(driver, bot):
    # For each iframe search for captchas
    iframe_list = []
    for iframe in driver.find_elements_by_tag_name("iframe"):
        iframe_list.append(iframe)

    while iframe_list:
        iframe = iframe_list.pop(0)
        driver.switch_to.frame(iframe)

        for iframe in driver.find_elements_by_tag_name("iframe"):
            iframe_list.append(iframe)

        element = driver.find_elements_by_id("recaptcha-anchor")[0]
        if element:
            break

    if element:
        # Esperamos a que el user este preparado
        bot.send_alert()
        while not bot.responses and "start" not in bot.responses:
            sleep(1)
        
        bot.responses.pop(0)
        
        # Lo ponemos visible
        driver.execute_script("arguments[0].scrollIntoView();", element)

        # usamos opencv para sacar la posicion en la pantalla
        img = pyautogui.screenshot()
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        # Find position
        template = cv2.imread('captcha.png',0)
        w, h = template.shape[::-1]

        # Apply template Matching and find captcha
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h-20)

        # We found the capcha position, now we move to some point inside the 
        point_inside = (np.random.randint(top_left[0],high=bottom_right[0]), np.random.randint(top_left[1],high=bottom_right[1]))
        mouse.move_to(point_inside)
        mouse.left_click()

        sleep(2)
        # Detect the boxes
        driver.switch_to.default_content()
        captcha_imageselect = driver.find_elements(By.XPATH, "/html/body/div/div[4]/iframe")[0]

        # Send captchas to user
        while True:
            bot.send_msg("Checking if captcha is done")
            sleep(2)
            # If we managed to solve it
            if driver.find_elements(By.CLASS_NAME, "recaptcha-checkbox-checkmark"):
                bot.send_msg("Success!")
                break
            
            # Check the type of captcha 
            captcha_type = 'notype'
            tile_size = 100 
            captcha_type = '3x3'
            tile_size = 127
            if driver.find_elements_by_class_name("rc-imageselect-table-44"):
                captcha_type = '4x4'
                tile_size = 95

            # Tomamos un screenshot de la pagina y lo cortamos para que sea el captcha solo
            driver.save_screenshot("img.png")
            img = cv2.imread("img.png", 1)

            location = captcha_imageselect.location
            size = captcha_imageselect.size
            x = int(location['x'])
            y = int(location['y'])
            w = int(size['width'])
            h = int(size['height'])

            img = img[y:y+h, x:x+w]
            cv2.imwrite('img.png', img)

            # Sacamos la posicion real del captcha
            screenshot = pyautogui.screenshot()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

            # Find position
            template = cv2.imread('img.png',0)
            w, h = template.shape[::-1]

            # Apply template Matching and find captcha
            res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)

            _, _, _, captcha_offset = cv2.minMaxLoc(res)

            # Send the captcha photo to the user
            with open('img.png', 'rb') as f:
                bot.send_captcha(f)

            # Wait for his input
            actions = []
            while True:
                if bot.responses:
                    action = bot.responses.pop(0)
                    if action[0] == 'do':
                        break
                    else:
                        actions.append(action)

            # Translate action into positions and do
            positions = []
            start_offset = np.array(captcha_offset, dtype=np.int32)

            for action in actions:
                if action[0] == "verify":
                    position_offset = np.array((292,530)) + start_offset
                    size = np.array((99,41))
                    pos = position_offset + np.int32(np.random.uniform(size=2) * size)
                    positions.append(pos)
                elif action[0] in ['start','do']:
                    continue
                else:
                    x_i = int(action[0])
                    y_i = int(action[1])
                    position_offset = np.array((tile_size*x_i, 127 + tile_size*y_i)) + start_offset
                    size = np.array((127,127))
                    pos = position_offset + np.int32(np.random.uniform(size=2) * size)
                    positions.append(pos)

            print(positions)

            mouse.chain_clicks(positions)

if __name__ == "__main__":
    # Start bot
    with open("config.json",'r') as f:
        config = json.load(f)

    token = config["token"]
    chat_id = config["chat_id"]
    url = "https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox.php" #config["test_url"]

    driver = webdriver.Firefox()
    t_bot = BotClient(token, chat_id)
    mouse = Mouse()

    #Remove navigator.webdriver Flag
    driver.execute_script("const newProto = navigator.__proto__; delete newProto.webdriver; navigator.__proto__ = newProto;")

    # Load webpage
    driver.maximize_window()
    driver.get(url)

    check_for_captchas(driver, t_bot)