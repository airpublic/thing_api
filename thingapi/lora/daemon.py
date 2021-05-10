from thingapi.lora.format import decode_payload
from thingapi.lora.listener import listen_forever
import pandas as pd
import logging


# These 4 LOC initialize Django so we can use the models
# DJANGO_SETTINGS_MODULE must be set
import sys, os
sys.path.append('.')
import django
django.setup()

# Now import Django stuff
from thingapi.models import RawLoraReadingV4


LOGGER_NAME = 'thinga.lora.daemon'


def connect_callback(success, client):
    logger = logging.getLogger(LOGGER_NAME)

    if success:
        logger.info("Successfully connected to TTN MQTT iface (LoRa)")
    else:
        logger.error("Could not connect to TTN MQTT iface (LoRa)")


def uplink_callback(msg, client):
    logger = logging.getLogger(LOGGER_NAME)

    try:
        logger.info("Received LORA packet from box %s" % msg.dev_id)

        record = decode_payload(msg.payload_raw)
        timestamp = pd.to_datetime(msg.metadata.time)
        device = int(msg.dev_id)

        reading = RawLoraReadingV4(timestamp=timestamp, id=device, **record._asdict())
        reading.save()

    except Exception as e:
        logger.exception(e)


def main():
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Starting LORA uplink daemon")
    listen_forever(connect_callback, uplink_callback)


if __name__ == '__main__':
    main()
