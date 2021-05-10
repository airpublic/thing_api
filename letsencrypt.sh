#!/bin/bash

# This runs on the host

/root/certbot/letsencrypt-auto certonly --webroot -w /root/thing_api/ -d thingapi.airpublic.eu  --agree-tos -m a.johncant@gmail.com -n --cert-path /root/thing_api/certs/thingapi.crt --key-path /root/thing_api/certs/thingapi.key
cp /etc/letsencrypt/live/thingapi.airpublic.eu/privkey.pem certs/thingapi.key
cp /etc/letsencrypt/live/thingapi.airpublic.eu/fullchain.pem certs/thingapi.crt

/root/certbot/letsencrypt-auto certonly --webroot -w /root/thing_api/ -d dashboards.airpublic.eu  --agree-tos -m a.johncant@gmail.com -n --cert-path /root/thing_api/certs/dashboards.crt --key-path /root/thing_api/certs/dashboards.key
cp /etc/letsencrypt/live/dashboards.airpublic.eu/privkey.pem certs/dashboards.key
cp /etc/letsencrypt/live/dashboards.airpublic.eu/fullchain.pem certs/dashboards.crt

/root/certbot/letsencrypt-auto certonly --webroot -w /root/thing_api/ -d '*.bokeh.airpublic.eu'  --agree-tos -m a.johncant@gmail.com -n --cert-path /root/thing_api/certs/bokeh.crt --key-path /root/thing_api/certs/bokeh.key
cp '/etc/letsencrypt/live/*.bokeh.airpublic.eu/privkey.pem' certs/wildcard.bokeh.key
cp '/etc/letsencrypt/live/*.bokeh.airpublic.eu/fullchain.pem' certs/wildcard.bokeh.crt
