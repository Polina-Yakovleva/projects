from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

def get_main_keyboard():
    inline_transfer = InlineKeyboardButton(text='Перенести стиль', callback_data='get_pic_1')
    inline_info = InlineKeyboardButton(text='Информация', callback_data='info')
    return InlineKeyboardMarkup(inline_keyboard=[[inline_transfer], [inline_info]])