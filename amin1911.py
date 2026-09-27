import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from openai import OpenAI
import asyncio
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import threading

ai_token=''

bot_token = '8429360617:AAEq7tbtVLbQ2P7Bx92vKW8-4gcnIW-mBGs'

logging.basicConfig( format='%(asctime)s - %(levelname)s - %(message)s',
    level = logging.INFO
)

logger = logging.getLogger(__name__)


class RuGPT3Bot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.load_model_async()

    def load_model_async(self):
        """Асинхронная загрузка модели в отдельном потоке"""

        def load_in_thread():
            try:
                logger.info("🔄 Загружаю RuGPT-3 модель...")

                # Модель RuGPT-3 от SberAI
                model_name = "sberbank-ai/rugpt3large_based_on_gpt2"

                # Загружаем токенайзер и модель
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                self.model = GPT2LMHeadModel.from_pretrained(model_name)

                # Добавляем специальные токены для русского языка
                self.tokenizer.add_special_tokens({
                    'pad_token': '[PAD]',
                    'eos_token': '</s>',
                    'bos_token': '<s>'
                })

                self.model_loaded = True
                logger.info("✅ RuGPT-3 модель успешно загружена!")

            except Exception as e:
                logger.error(f"❌ Ошибка загрузки модели: {e}")
                self.model_loaded = False

        # Запускаем загрузку в отдельном потоке
        thread = threading.Thread(target=load_in_thread)
        thread.daemon = True
        thread.start()

    async def generate_response(self, prompt):
        """Генерация ответа с помощью RuGPT-3"""
        if not self.model_loaded:
            return "🔄 Модель еще загружается... Попробуйте через минуту."

        try:
            # Подготавливаем промпт
            full_prompt = f"<s>{prompt}</s>"

            # Токенизируем входной текст
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt")

            # Генерируем ответ
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 100,  # Ограничиваем длину
                    num_return_sequences=1,
                    temperature=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )

            # Декодируем ответ
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Убираем оригинальный промпт из ответа
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            # Очищаем ответ
            response = self.clean_response(response)

            return response if response else "Не удалось сгенерировать ответ."

        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return "Извините, произошла ошибка при генерации ответа."

    def clean_response(self, text):
        """Очистка сгенерированного текста"""
        # Убираем лишние символы и обрезаем до точки
        text = text.replace('</s>', '').replace('<s>', '').strip()

        # Обрезаем до последней завершенной мысли
        if '.' in text:
            text = text[:text.rfind('.') + 1]
        elif '!' in text:
            text = text[:text.rfind('!') + 1]
        elif '?' in text:
            text = text[:text.rfind('?') + 1]

        return text


# Создаем экземпляр бота
rugpt_bot = RuGPT3Bot()


async def generate_resp(prompt, maxretries=5):
    """генерация с экспоненциальной задержкой при ошибках"""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=ai_token,
    )

    completion = client.chat.completions.create(
        model="x-ai/grok-4.1-fast:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    extra_body={"reasoning": {"enabled": True}}
    )
    completion = completion.choices[0].message.content
    messages = [
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": completion.content,
            "reasoning_details": completion.reasoning_details  # Pass back unmodified
        },
        {"role": "user", "content": prompt}
    ]

    # Second API call - model continues reasoning from where it left off
    response2 = client.chat.completions.create(
        model="x-ai/grok-4.1-fast:free",
        messages=messages,
        extra_body={"reasoning": {"enabled": True}}
    )
    return completion

async def start(update: Update, context:ContextTypes.DEFAULT_TYPE):
    '''функция старт'''
    welcome_text = """      
    Привет! Я бот с ИИ
    Мои команды:
     /start - начать работу
     /help - помощь
     /about - о боте
     /reset - сбросить диалог
     
     Напишите мне что-нибудь!
     """
    await update.message.reply_text(welcome_text)

    context.user_data['history']=[]
    context.user_data['user_name']=update.effective_user.first_name


async def help_command(update:Update, context:ContextTypes.DEFAULT_TYPE):
    '''функция помощь'''
    help_text = """Напишите мне и ИИ ответит! Как пользоваться:
    1. перессказ
    2. анализ персонажей
    3. помощь с рецензией
    """
    await update.message.reply_text(help_text)

async def about(update:Update, context:ContextTypes.DEFAULT_TYPE):
    '''функция о боте'''
    about_text = """Бот - литературный помощник с искусственным интелектом"""
    await update.message.reply_text(about_text)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Проверка статуса модели"""
    if rugpt_bot.model_loaded:
        status_text = "✅ Модель RuGPT-3 загружена и готова к работе!"
    else:
        status_text = "🔄 Модель еще загружается... Подождите 1-2 минуты."

    await update.message.reply_text(status_text)


async def reset_command(update:Update, context:ContextTypes.DEFAULT_TYPE):
    '''функция сброс диалога'''
    # очищаем историю разговооров
    if 'history' in context.user_data:
        context.user_data['history'] = []
    await update.message.reply_text('диалог сброшен. я все забыл ')


async def handlemes(update: Update, context:ContextTypes.DEFAULT_TYPE):
    '''обработка текстовых сообщений'''
    user_text = update.message.text
    user_name = update.effective_user.first_name

    logger.info(f'сообщение от {user_name}: {user_text}')

    # if contains_bad_words(user_text):
    #     await update.message.reply_text("прошу избегать нецензурных выражений")
    #     return
    #
    if len(user_text)>500:
        await update.message.reply_text("Сообщение слишком длинное, пожалуйста сократите до 500 символов: ")
        return



    await update.message.chat.send_action(action='typing')

    try:
        # prompt = context_prompt(user_text, context)
        if user_text.lower() in ['привет','здравствуй','добрый день', 'добрый вечер', 'доброе утро']:
            bot_response = f'Привет, {user_name}! Чем могу помочь?'
        elif user_text.lower() in ['как дела','как ты','как твое здоровье']:
            bot_response = "У меня отлично! Спасибо, что спросили!"
        else:
            bot_response = await generate_resp(user_text)

        # update_conversh(context, user_text, bot_response)

        await update.message.reply_text(bot_response)
        logger.info('Ответ успешно отправлен пользователю')
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        await update.message.reply_text("произошла непредвиденная ошибка.")


def generation_litresponse(user_text):
    textlower = user_text.lower()
    if any(word in textlower for word in ['привет','здравствуйте','hello']):
        return 'Привет! я Литературный помощник, чем могу помочь?'

    elif any(word in textlower for word in ['пересказ', "перескажи", "кратко", "краткий"]):
        return "Я готов сделать краткий перессказ произведения, пришлите название произведения или книги."
    else:
        return f"Вы написали: {user_text} \n\n Спасибо за интересный запрос, но, пожалуйста, используйте: \n - 'Перескажи [название]'\n -  ... "


def main():
    try:
        logger.info("Запуск бота...")
        application = Application.builder().token(bot_token).build()
        application.add_handler(CommandHandler('start', start))
        application.add_handler(CommandHandler('help', help_command))
        application.add_handler(CommandHandler('about', about))
        application.add_handler(CommandHandler('reset', reset_command))

        # обработка сообщений
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlemes))

        logger.info("бот запущен!")
        application.run_polling()

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        print("где-то пошла ошибка")


if __name__=="__main__":
    main()

