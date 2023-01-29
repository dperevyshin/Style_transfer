import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import os
import telebot
import Style_transfer
from telebot import  types

bot = telebot.TeleBot('5797371069:AAGPSzx7LQqA-j4r3T3fdd6of0WQT9uplr0')

device = "cuda" if torch.cuda.is_available() else 'cpu'
imsize = 512 if device == 'cuda' else 256
loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])

def image_loader(image_name):
  image = Image.open(image_name)
  print('????',image)
  image = loader(image).unsqueeze(0)
  return image.to(device, torch.float)

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Загрузи исходное изображение')
    bot.register_next_step_handler(message, get_content)

@bot.message_handler(content_types='photo')
def get_content(message):
    global content_img
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src_c = '/home/dmitrij/PycharmProjects/pythonProject/Style_transfer/' + message.photo[1].file_id + '.jpg'
    with open(src_c, 'wb') as new_file:
        new_file.write(downloaded_file)
    print('THIS',src_c)
    content_img = image_loader(src_c)
    os.remove(src_c)
    bot.reply_to(message, "Отлично, теперь загрузи стиль")
    bot.register_next_step_handler(message, get_style)


def get_style(message):
    global style_img
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src_s = '/home/dmitrij/PycharmProjects/pythonProject/Style_transfer/' + message.photo[1].file_id + '.jpg'
    with open(src_s, 'wb') as new_file:
        new_file.write(downloaded_file)
    style_img = image_loader(src_s)
    os.remove(src_s)
    bot.send_message(message.from_user.id, 'А теперь немного подожди')

    cnn = models.vgg19(pretrained = True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    output = Style_transfer.run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, content_img)
    save_image(output,"gen.png")
    photo = open('gen.png', 'rb')
    bot.send_photo(message.from_user.id, photo)
    keyboard = types.InlineKeyboardMarkup()
    key_yes = types.InlineKeyboardButton(text='Да', callback_data='yes')
    keyboard.add(key_yes)
    bot.send_message(message.from_user.id, 'Тебе понравился результат?', reply_markup=keyboard)
    os.remove('gen.png')

@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    if call.data == "yes":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton("/start")
        markup.add(btn1)
        bot.send_message(call.message.chat.id, ':)', reply_markup=markup);

bot.polling(none_stop=True, interval=0)