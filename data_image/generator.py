import random
import json
from PIL import Image
from PIL import ImageDraw, ImageFont


class RandomChar():

    @staticmethod
    def Unicode():
        val = random.randint(0x4E00, 0x9FBF)
        return unichr(val)

    @staticmethod
    def GB2312():
        head = random.randint(0xB0, 0xCF)
        body = random.randint(0xA, 0xF)
        tail = random.randint(0, 0xF)
        val = (head << 8) | (body << 4) | tail
        str = "%x" % val
        return str.decode('hex').decode('gb2312')


class ImageChar():
    def __init__(self, fontColor=(0, 0, 0),
                 size=(80, 40),
                 fontPath='simsun.ttc',
                 bgColor=(255, 255, 255),
                 fontSize=25):
        self.size = size
        self.fontPath = fontPath
        self.bgColor = bgColor
        self.fontSize = fontSize
        self.fontColor = fontColor
        self.font = ImageFont.truetype(self.fontPath, self.fontSize)
        self.image = Image.new('RGB', size, bgColor)

    def rotate(self):
        self.image.rotate(random.randint(0, 45), expand=0)

    def drawText(self, pos, txt, fill):
        draw = ImageDraw.Draw(self.image)
        draw.text(pos, txt, font=self.font, fill=fill)
        del draw

    def randRGB(self, mode='light'):
        if mode == 'light':
            return (random.randint(128, 255),
                    random.randint(128, 255),
                    random.randint(128, 255))
        elif mode == 'dark':
            return (random.randint(0, 128),
                    random.randint(0, 128),
                    random.randint(0, 128))

    def randPoint(self):
        (width, height) = self.size
        return (random.randint(0, width), random.randint(0, height))

    def randLine(self, num):
        draw = ImageDraw.Draw(self.image)
        for i in range(0, num):
            draw.line([self.randPoint(), self.randPoint()], self.randRGB('light'))
        del draw

    def randChinese(self, word, num_line):
        gap = 8
        num = len(word)
        start = 2
        for i in range(0, num):
            x = start + self.fontSize * i + random.randint(2, gap) + gap * i
            self.drawText((x, random.randint(0, 5)), word[i], self.randRGB('dark'))
            self.rotate()
        self.randLine(num_line)

    def save(self, path):
        self.image.save(path, 'png')


if __name__ == '__main__':
    with open('../data_text/common_words_2ch.txt', 'r') as fp:
        text = fp.read().split()
        data_info = open('./data_info.json', 'w')
        index_info = []
        import os
        os.mkdir('captcha_20')
        os.mkdir('captcha_40')
        os.mkdir('captcha_60')

        for i, word in enumerate(text):
            ic = ImageChar(fontColor=(200, 211, 170))
            ic.randChinese(word, 20)
            ic.save(f"./captcha_20/{i:04d}.png")
            index_info.append({'id': f'{i:04d}',
                               'label': word})

        for i, word in enumerate(text):
            ic = ImageChar(fontColor=(200, 211, 170))
            ic.randChinese(word, 40)
            ic.save(f"./captcha_40/{i:04d}.png")

        for i, word in enumerate(text):
            ic = ImageChar(fontColor=(200, 211, 170))
            ic.randChinese(word, 60)
            ic.save(f"./captcha_60/{i:04d}.png")

        json.dump(index_info, data_info, indent=4)
