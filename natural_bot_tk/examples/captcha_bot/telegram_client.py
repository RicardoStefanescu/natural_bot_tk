import json
import telebot
import threading
from time import sleep
from PIL import Image

class BotClient:
    def bot_loop(self):
        @self.bot.message_handler(commands=['help', 'start'])
        def send_welcome(message):
            self.bot.reply_to(message, "Sup jabroni")

        # Handle all other messages with content_type 'text' (content_types defaults to ['text'])
        @self.bot.message_handler(func=lambda message: True)
        def echo_message(message):
            self.bot.reply_to(message, "Shut up you shouldn't be talking")
        
        @self.bot.callback_query_handler(func=lambda call: True)
        def handle_query(call):
            if call.data == "start":
                self.responses.append("start")
            elif call.data == "send":
                self.responses.append("send")
            else:
                self.responses.append(call.data.split(","))

            print(f"Got : {call.data}")

        self.bot.polling()

    def __init__(self, token, chat_id):
        self.chat_id = chat_id
        self.bot = telebot.TeleBot(token)
        self.bot.skip_pending = True

        self.responses = []
        self.t = threading.Thread(target=self.bot_loop)
        self.t.start()

    def stop(self):
        self.t.join()

    def _make_keyboard(self):
        markup = telebot.types.InlineKeyboardMarkup()

        for i in range(3):
            row = []
            for j in range(3):
                row.append(telebot.types.InlineKeyboardButton(text=f"{j}-{i}", callback_data=f"{j},{i}"))
            markup.add(*row)

        markup.add(telebot.types.InlineKeyboardButton(text="Verify", callback_data="verify"))
        markup.add(telebot.types.InlineKeyboardButton(text="*Do actions*", callback_data="do"))
        return markup

    def send_msg(self, msg):
        self.bot.send_message(self.chat_id, msg)

    def send_alert(self):
        markup = telebot.types.InlineKeyboardMarkup()
        markup.add(telebot.types.InlineKeyboardButton(text="START", callback_data="start"))

        self.bot.send_message(self.chat_id, "Captcha found", reply_markup=markup)

    def send_captcha(self, img):
        self.bot.send_photo(self.chat_id, img, reply_markup=self._make_keyboard())

if __name__ == "__main__":
    with open("config.json",'r') as f:
        config = json.load(f)

    token = config["token"]
    chat_id = config["chat_id"]

    b = BotClient(token, chat_id)

    b.send_alert()
    with open('img.png', 'rb') as img:
        b.send_captcha(img)
