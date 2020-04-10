#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import cv2
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def save_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    # array = cv2.bilateralFilter(array, 5, 50, 50)
    return array


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def main():
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)

    world = client.get_world()
    client.set_timeout(2.0)
    i = 0
    os.mkdir("sun")
    os.mkdir("sun_sem")
    try:
        world = client.get_world()

        m = world.get_map()
        start_pose = carla.Transform(carla.Location(x=1, y=2, z=3), carla.Rotation())
        # start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.*')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(True)

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=5.5, z=2.5), carla.Rotation(pitch=8)),
            attach_to=vehicle)

        actor_list.append(camera_rgb)

        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=5.5, z=2.5), carla.Rotation(pitch=8)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)

        camera_depth = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=5.5, z=2.5), carla.Rotation(pitch=8)),
            attach_to=vehicle)
        actor_list.append(camera_depth)

        # weather1 = carla.WeatherParameters(cloudiness=80.0, precipitation=30.0, fog_density=0.0,
        #                                 sun_altitude_angle=-70.0)
        # world.render(display)
        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=30) as sync_mode:

            while True:

                if should_quit():
                    return
                world = client.get_world()
                clock.tick()
                # weather = carla.WeatherParameters(cloudiness=80.0, wind_intensity=0.0, fog_density=90.0, wetness=100.0,
                #                                  precipitation=100.0, precipitation_deposits=80.0,
                #                                  sun_altitude_angle=100.0)
                weather = carla.WeatherParameters(cloudiness=0.0, wind_intensity=0.0, fog_density=0.0, wetness=0.0,
                                                  precipitation=0.0, precipitation_deposits=0.0,
                                                  sun_altitude_angle=100.0)
                world.set_weather(weather)

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=0.3)

                # world.set_weather(weather1)w
                # snapshot1, image_rgb1, image_semseg = sync_mode.tick(tww

                # snapshot, image_rgb1, image_semseg, camera_depth = synww
                # Choose the next waypoint and update the car location.ww
                waypoint = random.choice(waypoint.next(0.5))
                vehicle.set_transform(waypoint.transform)
                # world.render(display)
                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                # fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # world.render()
                # Draw the display.
                # draw_image(display, image_rgb1)
                draw_image(display, image_rgb)
                image_rgb2 = save_image(image_semseg)
                image_rgb3 = save_image(image_rgb)
                cv2.imwrite("sun/out" + str(i) + ".png", image_rgb3)
                cv2.imwrite("sun_sem/out" + str(i) + ".png", image_rgb2)
                i += 1
                '''
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 5))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 18))
'''
                pygame.display.flip()

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
