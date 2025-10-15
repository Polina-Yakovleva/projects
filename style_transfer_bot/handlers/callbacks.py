from aiogram import F, types
from aiogram.fsm.context import FSMContext
from messages import MESSAGES
from handlers.images import UserState

async def get_pictures_process(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.answer(MESSAGES['send'])
    await callback.answer(MESSAGES['cb_answ'])
    await callback.message.delete()
    await state.set_state(UserState.picture1)

async def get_info_process(callback: types.CallbackQuery):
    await callback.message.answer(MESSAGES['info'])