#!/bin/bash

set -e

printenv | sed 's/^\(.*\)$/export \1/g' > /cron_env.sh
chmod +x /cron_env.sh
touch /etc/crontab /etc/cron.*/* && cron


for i in $(seq 0 $(($BOKEH_NUM_PROCESSES-1)))
do

if [ -z "$DEV" ]
then
  bokeh serve experiment\
    --port "$((5000+$i))"\
    --allow-websocket-origin 'thingapi.airpublic.local'\
    --allow-websocket-origin 'thingapi.airpublic.eu'\
    --allow-websocket-origin 'dashboards.airpublic.eu'\
    --use-xheaders\
    --unused-session-lifetime 300000\
    >> bokeh.stdout\
    2>> bokeh.stderr\
    &
else
  bokeh serve experiment\
    --dev experiment/*\
    --port "$((5000+$i))"\
    --allow-websocket-origin 'localhost:8000'\
    --allow-websocket-origin "localhost:$((5000+$i))"\
    --allow-websocket-origin "thingapi-internal.airpublic.local:$((5000+$i))"\
    --allow-websocket-origin 'thingapi-internal.airpublic.local:8001'\
    --allow-websocket-origin 'thingapi.airpublic.local'\
    --allow-websocket-origin "bokeh-$i.bokeh.airpublic.local"\
    --use-xheaders\
    --unused-session-lifetime 300000\
    --log-level=debug \
    >> bokeh.stdout\
    2>> bokeh.stderr\
    &
fi
done


python -m thingapi.lora.daemon >> lora.log 2>> lora.err &


gunicorn thingapi.wsgi:application \
  --reload\
  --worker-class sync\
  --name thingapi_django\
  --bind 0.0.0.0:8000\
  --workers 20\
  --threads 1\
  --log-level=info\
  --log-file=gunicorn.log\
  --access-logfile=gunicorn.access.log\
  &

tail -f \
  cron.err\
  cron.out\
  bokeh.stdout\
  bokeh.stderr\
  gunicorn.log\
  gunicorn.access.log\
  lora.err\
  lora.log\

