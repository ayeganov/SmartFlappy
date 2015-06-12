#!/usr/bin/env python

__version__ = '1.0'

import itertools
import random

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.core.audio import SoundLoader

sfx_flap = SoundLoader.load('audio/flap.wav')
sfx_score = SoundLoader.load('audio/score.wav')
sfx_die = SoundLoader.load('audio/die.wav')

from config import config
import PID


class Sprite(Image):
    def __init__(self, **kwargs):
        super(Sprite, self).__init__(**kwargs)
        self.size = self.texture_size

class Menu(Widget):
    def __init__(self):
        super(Menu, self).__init__()
        self.add_widget(Sprite(source='images/background.png'))
#        self.size = self.children[0].size
        self.size = (200, 1820)
        self.add_widget(Ground(source='images/ground.png'))
        self.add_widget(Label(center=self.center, text="Tap to Start"))

    def on_touch_down(self, *ignore):
        parent = self.parent
        parent.remove_widget(self)
        parent.add_widget(Game())


class Bird(Sprite):
    def __init__(self, pos):
        super(Bird, self).__init__(source='atlas://images/bird_anim/wing-up', pos=pos)
        min_vel = config['min_velocity']
        max_vel = config['max_velocity']

        self._pid = PID.PID(KP=13.197,
                            KI=0.0,
                            KD=0.256,
                            min_cor=min_vel,
                            max_cor=max_vel)

        self.velocity_y = 0
        self._gravity = config['gravity']
        self._flap_images = itertools.cycle(('atlas://images/bird_anim/wing-up',
                            'atlas://images/bird_anim/wing-mid',
                            'atlas://images/bird_anim/wing-down',
                            'atlas://images/bird_anim/wing-mid'))
        self._glide_image = 'atlas://images/bird_anim/wing-mid'

    def update(self, dt):
        self.velocity_y += (self._gravity * dt)
        self.y += (self.velocity_y * dt)

#        print "Velocity: {0}, position: {1}".format(self.velocity_y, self.y)
        if abs(self.velocity_y) > 3:
            self.source = next(self._flap_images)
        else:
            self.source = self._glide_image

    def fly_to(self, height):
        self._pid.target = height

    def flap(self, passed_time):
        '''
        Increase, or decrease power of flapping of the flappy bird.

        @param passed_time - amount of time passed since last flap
        '''
        self.velocity_y = self._pid.make_correction(self.y, passed_time)

    def on_touch_down(self, touch):
#        self.velocity_y = 2.5
        self.source = 'atlas://images/bird_anim/wing-down'
        sfx_flap.play()

        x, y = touch.pos
        self.fly_to(y)


class Background(Widget):
    def __init__(self, source):
        super(Background, self).__init__()
        self.image = Sprite(source=source)
        self.add_widget(self.image)
        self.size = self.image.size
        self.image_dupe = Sprite(source=source, x=self.width)
        self.add_widget(self.image_dupe)

    def update(self):
        self.image.x -= 2
        self.image_dupe.x -= 2

        if self.image.right <= 0:
            self.image.x = 0
            self.image_dupe.x = self.width


class Ground(Sprite):
    def update(self):
        self.x -= 2
        if self.x < -24:
            self.x += 24


class Pipe(Widget):
    def __init__(self, pos):
        super(Pipe, self).__init__(pos=pos)
        self.top_image = Sprite(source='images/pipe_top.png')
        self.top_image.pos = (self.x, self.y + 3.5 * 24)
        self.add_widget(self.top_image)

        self.bottom_image = Sprite(source='images/pipe_bottom.png')
        self.bottom_image.pos = (self.x, self.y - self.bottom_image.height)
        self.add_widget(self.bottom_image)
        self.width = self.top_image.width
        self.scored = False

    def update(self):
        self.x -= 2
        self.top_image.x = self.bottom_image.x = self.x
        if self.right < 0:
            self.parent.remove_widget(self)


class Pipes(Widget):
    add_pipe = 0

    def update(self, dt):
        for child in list(self.children):
            child.update()

        self.add_pipe -= dt
        if self.add_pipe < 0:
            y = random.randint(self.y + 50, self.height - 50 - 3.5 * 24)
            self.add_widget(Pipe(pos=(self.width, y)))
            self.add_pipe = 1.5


class Game(Widget):
    def __init__(self):
        super(Game, self).__init__()
        self.background = Background(source='images/background.png')
        self.size = self.background.size 
        self.add_widget(self.background)

        self.bird = Bird(pos=(50, self.height / 2))
        self.bird.fly_to(100)
        self.add_widget(self.bird)

        self.ground = Ground(source='images/ground.png')
        self.pipes = Pipes(pos=(0, self.ground.height), size=self.size)
        self.add_widget(self.pipes)

        self.add_widget(self.ground)
        self.score_label = Label(center_x=self.center_x, top=self.top - 30, text='0')
        self.add_widget(self.score_label)
        self.over_label = Label(center=self.center, opacity=0, text="Game over!")
        self.add_widget(self.over_label)

        Clock.schedule_interval(self.update, 1.0/config['fps'])
        self.game_over = False
        self.score = 0

    def update(self, dt):
        if self.game_over:
            Clock.unschedule(self.update)

#        self.background.update()
        self.bird.update(dt)
        self.bird.flap(dt)

#        self.ground.update()

#        self.pipes.update(dt)

#        if self.bird.collide_widget(self.ground):
#            self.game_over = True

        for pipe in self.pipes.children:
            if pipe.top_image.collide_widget(self.bird):
                self.game_over = True
                print("Bump" + str(random.randint(0, 10)))
            elif pipe.bottom_image.collide_widget(self.bird):
                self.game_over = True
                print("Bump" + str(random.randint(0, 10)))
            elif not pipe.scored and pipe.right < self.bird.x:
                pipe.scored = True
                self.score += 1
                self.score_label.text = str(self.score)
                sfx_score.play()

        if self.game_over:
            sfx_die.play()
            self.over_label.opacity = 1
            self.bind(on_touch_down=self._on_touch_down)

    def _on_touch_down(self, *ignore):
        parent = self.parent
        parent.remove_widget(self)
        parent.add_widget(Menu())

class GameApp(App):
    def build(self):
        top = Widget()
        top.add_widget(Menu())
        Window.size = top.children[0].size
        return top


def main():
    GameApp().run()

if __name__ == "__main__":
    main()
