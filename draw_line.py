import pygame as pg
import AI
from AI import load_model
from PIL import Image , ImageOps
from tensorflow.image import per_image_standardization
import numpy as np

pg.init()

WHITE = (255,255,255)
BLACK = (0,0,0)

screen_width = 500
screen_height = 500

screen = pg.display.set_mode((screen_width,screen_height))
screen.fill(WHITE)



output_path = "images/screenshot.png"
resize_path = "images/screenshot_resize.png"

def crope(screen):
    cropped = pg.Surface((screen_width-5,screen_height-5))
    cropped.blit(screen,(0,0),(0,0,screen_width-5,screen_height-5))
    return cropped

def image_processing(image_path):
    img = Image.open(image_path)
    new_img = img.resize((28,28))
    new_img = ImageOps.grayscale(new_img)
    new_img_array = np.array(new_img)
    new_img_array_reshaped = np.reshape(new_img_array,(784))
    new_img_norm = (new_img_array_reshaped - np.mean(new_img_array_reshaped)) / (np.std(new_img_array_reshaped))
    new_img = np.reshape(new_img_norm,(1,784))
    return new_img
    


model = load_model.load_model()


LEFT = 1
RIGHT = 3

running = True

start_pos = None
drawing = False

while running:

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == LEFT:
                start_pos = event.pos
                drawing = True
            elif event.button == RIGHT:
                screen.fill(WHITE)
        if event.type == pg.MOUSEBUTTONUP:
            if event.button == LEFT:
                start_pos = None
                drawing = False
                img = crope(screen)
                pg.image.save(img,output_path)
                new_img = image_processing(output_path)
                prediction = np.argmax(model.predict(new_img))
                print(prediction)
        if event.type == pg.MOUSEMOTION:
            if drawing:
                mouse_pos = pg.mouse.get_pos()
                if start_pos != None:
                    pg.draw.line(screen,BLACK,start_pos,mouse_pos,15)
                start_pos = mouse_pos

    pg.display.update()