import os
import subprocess
from aiogram import F, types
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile
from config import bot, PICTURES_DIR
from messages import MESSAGES
from keyboards import get_main_keyboard

class UserState(StatesGroup):
    reg = State()
    picture1 = State()
    picture2 = State()

async def get_picture_1(message: types.Message, state: FSMContext):
    file_path = f"{PICTURES_DIR}/{message.from_user.id}_1.jpg"
    await bot.download(message.photo[-1], destination=file_path)
    await message.answer(MESSAGES['pic1'])
    await state.set_state(UserState.picture2)

async def get_picture_2(message: types.Message, state: FSMContext):
    file_path = f"{PICTURES_DIR}/{message.from_user.id}_2.jpg"
    await bot.download(message.photo[-1], destination=file_path)
    await message.answer(MESSAGES['pic2'])
    
    input_path = f"{PICTURES_DIR}/{message.from_user.id}_1.jpg"
    style_path = f"{PICTURES_DIR}/{message.from_user.id}_2.jpg"
    output_path = f"{PICTURES_DIR}/{message.from_user.id}_result.jpg"
    
    processing_msg = await message.answer("🔄 Обрабатываю изображения нейросетью...")
    
    try:
        result = subprocess.run([
            'style_transfer', 
            input_path, 
            style_path, 
            '-o', output_path,
            '--devices', 'cuda:0'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            await bot.delete_message(chat_id=message.chat.id, message_id=processing_msg.message_id)
            
            try:
                with open(output_path, 'rb') as file:
                    photo_data = file.read()
                photo = BufferedInputFile(photo_data, filename="result.jpg")
                await message.answer_photo(photo, caption=MESSAGES['result'])
                
            except Exception as photo_error:
                await message.answer(f"❌ Ошибка при отправке фото: {photo_error}")
            
            for file_path in [input_path, style_path, output_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
        else:
            error_msg = result.stderr if result.stderr else "Неизвестная ошибка"
            await message.answer(f"❌ Ошибка при переносе стиля: {error_msg}")
            
    except subprocess.TimeoutExpired:
        await message.answer("⏰ Превышено время обработки. Попробуйте с меньшими изображениями.")
    except Exception as e:
        await message.answer(f"❌ Ошибка: {str(e)}")
    
    await message.answer(MESSAGES['begin'])
    await state.set_state(UserState.reg)