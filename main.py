from typing import Callable, Any, Tuple, List
from collections import namedtuple
from dataclasses import dataclass
from numpy.ma import arange
from copy import deepcopy
from random import randint
import time
import math
import os
import pygame
from pygame.locals import *


@dataclass
class Limits:
    min: Any
    max: Any

    def __iter__(self):
        return (x for x in [self.min, self.max])


@dataclass
class Axis:
    re: Limits
    im: Limits

    def __iter__(self):
        return (x for x in [self.re, self.im])


@dataclass
class Scale:
    x: Callable
    y: Callable


Pos = Corners = namedtuple('Corners', 'x y')


def make_in_bounds(value, default_cond: List[Tuple[Any, Callable[[Any], bool]]]):
    for default, cond in default_cond:
        if not cond(value):
            return default
    return value


class Mandelbrot:
    def __init__(self, width, height):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 45)
        pygame.init()
        self.is_game_work: bool = True

        self.screen_sizes = namedtuple('Sizes', 'width height')(width, height)
        self.window_surface: pygame.Surface = pygame.display.set_mode(self.screen_sizes)
        self.text_font: pygame.font.Font = pygame.font.SysFont('Comic Sans MS', 15)

        self.cartesian_area = Axis(re=Limits(-2.0, 1.0), im=Limits(complex(0, -1), complex(0, 1)))
        self.temp_cartesian_area = None
        self.move_pos_start = None
        self.scales = self.calc_scales()

        self.zoom_scale = 0.8
        self.zoom_rect: pygame.Rect = self.make_zoom_rect()
        self.zoom_rect_corners = self.calc_corners(self.zoom_rect)
        self.last_zoom_change = time.monotonic()

        self.is_update_graphs = True
        self.f = pygame.Surface(self.screen_sizes)
        self.grid = None

        self.keys_state = dict()

    def main_loop(self):
        clock = pygame.time.Clock()

        while self.is_game_work:
            clock.tick(60)
            self.handle_input()
            self.update()
            self.draw()
        pygame.quit()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.is_game_work = False

            elif event.type == KEYDOWN and event.key in [K_KP_PLUS, K_KP_MINUS]:
                self.keys_state[event.key] = True

            elif event.type == KEYUP and event.key in [K_KP_PLUS, K_KP_MINUS]:
                self.keys_state[event.key] = False

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == BUTTON_LEFT:
                    self.keys_state[BUTTON_LEFT] = True
                    self.temp_cartesian_area = deepcopy(self.cartesian_area)
                    self.move_pos_start = event.pos

            elif event.type == MOUSEBUTTONUP:
                if event.button == BUTTON_WHEELDOWN:
                    self.zoom_out_cartesian_area()

                elif event.button == BUTTON_WHEELUP:
                    self.zoom_in_cartesian_area()

                elif event.button == BUTTON_LEFT:
                    self.keys_state[BUTTON_LEFT] = False

            elif event.type == MOUSEMOTION and self.keys_state.get(BUTTON_LEFT):
                self.move_cartesian_area(self.calc_move_vector(Pos(*event.pos), Pos(*self.move_pos_start)))

        for key, is_pressed in self.keys_state.items():
            if is_pressed and key == K_KP_PLUS and time.monotonic() - self.last_zoom_change > 0.01:
                self.last_zoom_change = time.monotonic()
                self.change_zoom_scale(-0.005)
            elif is_pressed and key == K_KP_MINUS and time.monotonic() - self.last_zoom_change > 0.01:
                self.last_zoom_change = time.monotonic()
                self.change_zoom_scale(+0.005)

    def update(self):
        self.scales = self.calc_scales()
        self.zoom_rect = self.make_zoom_rect()
        self.zoom_rect_corners = self.calc_corners(self.zoom_rect)
        if self.is_update_graphs:
            self.calc_mandelbrot()
            # self.grid = self.make_grid()
            self.is_update_graphs = False

    def calc_move_vector(self, curr_pos, prev_pos):
        return namedtuple('DIRECTION', 'x y')(
            -(self.scales.pixel_to_coord.x(curr_pos.x) - self.scales.pixel_to_coord.x(prev_pos.x)),
            -(self.scales.pixel_to_coord.y(curr_pos.y) - self.scales.pixel_to_coord.y(prev_pos.y))
        )

    def move_cartesian_area(self, move_vector):
        self.cartesian_area.re.min = self.temp_cartesian_area.re.min + move_vector.x
        self.cartesian_area.re.max = self.temp_cartesian_area.re.max + move_vector.x
        self.cartesian_area.im.min = self.temp_cartesian_area.im.min + complex(0, move_vector.y)
        self.cartesian_area.im.max = self.temp_cartesian_area.im.max + complex(0, move_vector.y)
        self.is_update_graphs = True

    def change_zoom_scale(self, value):
        self.zoom_scale += value
        self.zoom_scale = make_in_bounds(self.zoom_scale, [
            (0, lambda v: v >= 0),
            (1, lambda v: v <= 1)
        ])

    def zoom_in_cartesian_area(self):
        p2c = self.scales.pixel_to_coord
        re, im = self.cartesian_area
        zrc = self.zoom_rect_corners
        # Replace area sizes with zoom_rect sizes
        re.min = p2c.x(zrc.x.min)
        re.max = p2c.x(zrc.x.max)
        im.min = complex(0, p2c.y(zrc.y.max))
        im.max = complex(0, p2c.y(zrc.y.min))
        self.is_update_graphs = True

    def zoom_out_cartesian_area(self):
        p2c = self.scales.pixel_to_coord
        re, im = self.cartesian_area
        zrc = self.zoom_rect_corners
        x_min, x_max = p2c.x(zrc.x.min), p2c.x(zrc.x.max)
        y_min, y_max = p2c.y(zrc.y.max), p2c.y(zrc.y.min)
        # Increase area sizes via diffs with zoom_rect
        re.min = x_min - 2*abs(re.min - x_min)
        re.max = x_max + 2*abs(re.max - x_max)
        im.min = complex(0, y_min - 2*abs(im.min.imag - y_min))
        im.max = complex(0, y_max + 2*abs(im.max.imag - y_max))
        self.is_update_graphs = True

    def calc_scales(self):
        re, im = deepcopy(self.cartesian_area)
        width, height = self.screen_sizes

        def coord_to_pixel_x(x):
            x = make_in_bounds(x, [
                (re.min, lambda v: v >= re.min),
                (re.max, lambda v: v <= re.max)
            ])
            return int((x - re.min) / (re.max - re.min) * width)

        def coord_to_pixel_y(y):
            y = make_in_bounds(y, [
                (im.min.imag, lambda v: v >= im.min.imag),
                (im.max.imag, lambda v: v <= im.max.imag)
            ])
            return height - int((y - im.min.imag) / (im.max.imag - im.min.imag) * height)

        def pixel_to_coord_px(px):
            return (px - 0) / (width - 0) * (re.max - re.min) + re.min

        def pixel_to_coord_py(py):
            return (height - py - 0) / (height - 0) * (im.max.imag - im.min.imag) + im.min.imag

        return namedtuple('Scales', ['coord_to_pixel', 'pixel_to_coord'])(
            Scale(coord_to_pixel_x, coord_to_pixel_y),
            Scale(pixel_to_coord_px, pixel_to_coord_py)
        )

    def make_zoom_rect(self):
        width = int(self.zoom_scale * self.screen_sizes.width)
        height = int(self.zoom_scale * self.screen_sizes.height)

        x, y = pygame.mouse.get_pos()
        rect_pos_x = x - width // 2
        rect_pos_y = y - height // 2

        return Rect([rect_pos_x, rect_pos_y, width, height])

    @staticmethod
    def calc_corners(rect: pygame.Rect):
        x_min = rect.left
        x_max = rect.left + rect.width
        y_min = rect.top
        y_max = rect.top + rect.height
        return Corners(
            Limits(x_min, x_max),
            Limits(y_min, y_max)
        )

    def render_number(self, x):
        if isinstance(x, int):
            x = float(x)
        text: pygame.Surface = self.text_font.render('{:.5}'.format(x), True, (0, 0, 0))
        background = pygame.Surface(text.get_size())
        background.fill((255, 255, 255, 255))
        background.blit(text, (0, 0))
        return background

    @staticmethod
    def calc_text_offsets(text_sizes, pos: Tuple[Any, Any], c: Corners):
        t_width, t_height = text_sizes
        x, y = pos
        offset_x = (x - c.x.min) / (c.x.max - c.x.min + 1) * t_width
        offset_y = (y - c.y.min) / (c.y.max - c.y.min + 1) * t_height
        return namedtuple('TextOffset', 'x y')(offset_x, offset_y)

    def generate_rendered_numbers(self):
        p2c = self.scales.pixel_to_coord
        re, im = self.cartesian_area
        w, h = self.screen_sizes
        zrc = self.zoom_rect_corners
        src = self.calc_corners(Rect(0, 0, w, h))
        zcx, zcy = (zrc.x.min + zrc.x.max)/2, (zrc.y.min + zrc.y.max)/2
        numbers_to_render = [
            [re.min, [0, h/2], src],
            [re.max, [w, h/2], src],
            [im.min, [w/2, h], src],
            [im.max, [w/2, 0], src],
            [p2c.x(zrc.x.min), [zrc.x.min,       zcy], zrc],
            [p2c.x(zrc.x.max), [zrc.x.max,       zcy], zrc],
            [p2c.y(zrc.y.min), [zcx,       zrc.y.min], zrc],
            [p2c.y(zrc.y.max), [zcx,       zrc.y.max], zrc]
        ]
        for i, [number, [x, y], corners] in enumerate(numbers_to_render):
            rendered_number = numbers_to_render[i][0] = self.render_number(number)
            offsets = self.calc_text_offsets(rendered_number.get_size(), (x, y), corners)
            numbers_to_render[i][1] = (int(x - offsets.x), int(y - offsets.y))

        return [(rendered, pos) for rendered, pos, _ in numbers_to_render]

    def is_in_mandelbrot(self, px, py):
        z = complex(0)
        c = complex(self.scales.pixel_to_coord.x(px), self.scales.pixel_to_coord.y(py))

        is_in_set = True
        max_iterations = 50
        for i in range(max_iterations):
            next_z = z**2 + c
            if abs(next_z) > 2:
                is_in_set = False
                break
            else:
                z = next_z

        return is_in_set

    def calc_mandelbrot(self):
        black = (0, 0, 0)
        white = (255, 255, 255)
        pixels_view = pygame.PixelArray(self.f)
        for py in range(self.screen_sizes.height):
            for px in range(self.screen_sizes.width):
                if self.is_in_mandelbrot(px, py):
                    pixels_view[px, py] = black
                else:
                    pixels_view[px, py] = white
        del pixels_view

    def calc_parabola(self):
        f_surface = pygame.Surface(self.screen_sizes)
        f_surface.fill((255, 255, 255, 0))
        start, stop = self.cartesian_area.re
        for x in arange(start, stop, (stop - start)/2000):
            px = self.scales.coord_to_pixel.x(x)
            py = self.scales.coord_to_pixel.y(x**2)
            f_surface.fill((0, 0, 0), pygame.Rect((px, py), (1, 1)))
        return f_surface

    def make_grid(self):
        grid = pygame.Surface(self.screen_sizes, pygame.SRCALPHA)
        grid.fill((255, 255, 255, 0))

        re, im = self.cartesian_area
        c2p = self.scales.coord_to_pixel

        # Draw horizontal line
        start_line = (c2p.x(re.min), c2p.y(0))
        end_line = (c2p.x(re.max), c2p.y(0))
        pygame.draw.line(grid, (0, 0, 0), start_line, end_line, 3)

        # Draw vertical line
        start_line = (c2p.x(0), c2p.y(im.min.imag))
        end_line = (c2p.x(0), c2p.y(im.max.imag))
        pygame.draw.line(grid, (0, 0, 0), start_line, end_line, 3)

        return grid

    def draw(self):
        self.window_surface.blit(self.f, (0, 0))
        # self.window_surface.blit(self.grid, (0, 0))
        pygame.draw.rect(self.window_surface, (0, 0, 0), self.zoom_rect, 2)
        # for number, pos in self.generate_rendered_numbers():
        #     self.window_surface.blit(number, pos)
        pygame.display.flip()


# Technical part
def main():
    Mandelbrot(1024, 512).main_loop()


if __name__ == '__main__':
    main()
