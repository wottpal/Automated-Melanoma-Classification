# -*- coding: utf-8 -*-
"""Helper to send Notifications via [Prowl](www.prowlapp.com)."""

from . import config
import requests
import socket
import keras
import time


PROWL_API_KEY = config.get("PROWL_API_KEY")
PROWL_API_URL = "https://api.prowlapp.com/publicapi/add"
HOST_NAME = socket.gethostname()


def send_message(event, description = "", priority = 0):
    """Sends a notification via Prowl with the given event-name and description."""

    r = requests.post(PROWL_API_URL, data = {
        'apikey' : PROWL_API_KEY,
        'application' : HOST_NAME,
        'event' : event,
        'priority' : priority,
        'description' : description
    })

    if r.status_code is not 200:
        print(f"Could not send Prowl-Notification ({r.status_code}, {r.reason})")
        print(r.text)


class NotificationCallback(keras.callbacks.Callback):
    
    def prettify_logs(self, logs):
        return f"loss: {logs['loss']:.2f} - acc: {logs['acc']:.2f} - v_loss: {logs['val_loss']:.2f} - v_acc: {logs['val_acc']:.2f}"

    def on_train_begin(self, logs={}):
        self.start_time = time.perf_counter()
        send_message('🏃‍ Start Training 🏃‍', '', priority=1)

    def on_epoch_end(self, epoch, logs={}):
        send_message(f'Epoch {epoch + 1}', self.prettify_logs(logs))

    def on_train_end(self, logs={}):
        duration_min = int((time.perf_counter() - self.start_time) / 60)
        duration_sec = int(time.perf_counter() - self.start_time)
        duration = f'{duration_min} min' if duration_min else f'{duration_sec} sec' 
        send_message(f'🏁 Finished Training 🏁 ({duration})', priority=1)
