import logging
import asyncio
import os
from aiogram import Dispatcher, F
from aiogram.filters import Command

from config import bot, storage, PICTURES_DIR
from handlers.start import start_command, help_command, handle_reg_state
from handlers.callbacks import get_pictures_process, get_info_process
from handlers.images import get_picture_1, get_picture_2, UserState

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Создание диспетчера
dp = Dispatcher(storage=storage)

# Регистрация обработчиков команд
dp.message.register(start_command, Command("start"))
dp.message.register(help_command, Command("help"))

# Регистрация обработчиков состояний
dp.message.register(handle_reg_state, UserState.reg)
dp.message.register(get_picture_1, UserState.picture1, F.content_type == 'photo')
dp.message.register(get_picture_2, UserState.picture2, F.content_type == 'photo')

# Регистрация обработчиков callback-запросов
dp.callback_query.register(get_pictures_process, F.data == 'get_pic_1', UserState.reg)
dp.callback_query.register(get_info_process, F.data == 'info', UserState.reg)

async def main():
    print("🤖 Бот запускается...")
    # Создаем директорию для картинок
    os.makedirs(PICTURES_DIR, exist_ok=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())