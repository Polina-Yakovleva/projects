from aiogram import F, types
from aiogram.fsm.context import FSMContext
from aiogram.filters import Command
from messages import MESSAGES
from keyboards import get_main_keyboard
from handlers.images import UserState

async def start_command(message: types.Message, state: FSMContext):
    await message.answer(MESSAGES['start'])
    await message.answer(MESSAGES['begin'])
    await state.set_state(UserState.reg)

async def help_command(message: types.Message):
    await message.answer(MESSAGES['info'])

async def handle_reg_state(message: types.Message):
    await message.answer(MESSAGES['keyboard'], reply_markup=get_main_keyboard())