import time
import ttn
from time import sleep
# This doesn't import most of Django, so this module does not really depend on django
from django.conf import settings


def listen_forever(connect_callback, uplink_callback):
    handler = ttn.HandlerClient(settings.TTN_APP_ID, settings.TTN_ACCESS_KEY)

    # using application manager client
    app_client =  handler.application()

    my_devices = app_client.devices()

    # using mqtt client
    mqtt_client = handler.data()

    mqtt_client.set_connect_callback(connect_callback)
    mqtt_client.set_uplink_callback(uplink_callback)

    mqtt_client.connect()
    mqtt_client.start()

    # mqtt_client has no interface for getting status or for getting thread
    # or for waiting forever. But the underlying paho one does. We avoid using
    # that because the ttn wrapper supports reconnecting if the paho MQTT
    # client disconnects

    while(True):
        sleep(1)

    # This will never be run
    mqtt_client.close()
