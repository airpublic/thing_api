# Airpublic thingapi

This codebase is responsible for accepting data from our things, out there on
the internet. It also has an admin interface, dashboards, and various other features.

## Getting started

##### 1. Set up containers

```bash
docker-compose up
```

##### 2. Set up database

Run `docker exec -it postgres su postgres -c psql` and enter the
following SQL:

```SQL
CREATE DATABASE airpublic;
CREATE USER airpublic WITH PASSWORD 'c57029da5d';
GRANT ALL PRIVILEGES ON DATABASE airpublic TO airpublic;
```

Next step is to create the tables. You can do this by either running the migrations or restoring from a backup. If the backup is behind the current migration state, you can bring it up-to-date by running the migrations afterwards.

Dumping a recent production database is a good idea if you want to use
experiments that have been set up.

To take a backup of current production database:
```bash
pg_dump postgres://username:password@host:port/database > dump.sql
```

To restore from backup:
```bash
docker exec -i thingapi python manage.py dbshell < dump.sql
```

To run migrations
```bash
python manage.py migrate
```

##### 3. Done

Your development environment is now ready

## Useful commands

Get into the Postgres database currently being used
```bash
docker exec -it thingapi python manage.py dbshell
```

Quickly get an IPython console with Django already initialized. If the shell is just Python and not IPython, you need to install IPython
```bash
docker exec -it thingapi python manage.py shell
```

Get a bash console up
```bash
docker exec -it thingapi /bin/bash
```

Generate an experiment for all v4 data or all v3 data. In an IPython shell:
```python
import thingapi.models
thingapi.models.create_all_data_experiment_v3()
thingapi.models.create_all_data_experiment_v4()
```

Create a superuser if you can't access the admin:
```bash
docker exec -it thingapi python manage.py createsuperuser
```

Replace the Bokeh server with your own, so you can debug it and keep restarting it:
```bash
# Run everything here on the thingapi container

# Find the bokeh server
ps -ef | grep bokeh

# Kill bokeh. This will not kill the container
kill # <bokeh PID>

# Start your own bokeh server
# I (JC) put the following in a file called bokeh.sh

#!/bin/bash
bokeh serve experiment\
  --port 5006\
  --allow-websocket-origin 'localhost:8000'\
  --allow-websocket-origin 'localhost:5006'\
  --allow-websocket-origin '172.20.0.4:5006'\
  --allow-websocket-origin 'thingapi-internal.airpublic.local:5006'\
  --allow-websocket-origin 'thingapi-internal.airpublic.local:8000'\
  --allow-websocket-origin 'thingapi.airpublic.local'\
  --allow-websocket-origin 'bokeh-0.bokeh.airpublic.local'\
  --use-xheaders

```

To develop the Django part, no action is required since with the default settings in this repo, Django should watch for changes and reload the code. However, you can start your own Django server listening on its own port. You do not have to kill `GUnicorn` to do this.
```bash
docker exec -it thingapi python manage.py runserver 0.0.0.0:choosesomeport
```

## Container architecture
`docker-compose up` sets up 4 containers:

  - `postgres` - the database container. Data is on a shared volume so it is preserved on termination. In production we do not use this and in fact run PostgresQL on the host. Sometimes I have noticed the database gets wiped under the wrong conditions during development, i.e. checking out commits from before we implemented this, so please don't rely too much on it.
  - `thingapi` - the app servers. This runs `thingapi` under GUnicorn, and a Bokeh server. This is where migrations should be run from.
  - `nginx` - the web server. This deals with SSL termination and reverse proxies to thingapi. It should also be set up to serve static files.
  - `dns` - a DNS proxy. This hijacks your /etc/resolv.conf so you can address docker containers using hostnames instead of IP addresses. We don't use this in production. Please make sure you are ok this before running, since it intercepts all your DNS queries. This container can sometimes break your internet, especially if you switch WiFi networks. Sometimes restarting this container with `docker-compose restart dns` is sufficient, but if not, then killing it always fixes the problem.
  
There is no built-in way to launch jupyter notebook so you should just do this in your own way.

## Bokeh embedding

The way we've embedded Bokeh is complicated and unsupported by the Bokeh developers.

1. A user visits https://dashboards.airpublic.eu/experiments/1/ or you visit https://thingapi.airpublic.local/experiments/1/
2. The request is proxied by nginx through to Django
3. Our Django app uses bokeh's supplied bokeh client to make a request to Bokeh to retrieve a serialized version of the session. The `pull_session` function call does this in `thingapi.views` function `experiment`. We use the environment variable `BOKEH_INTERNAL_ROOT` to know which URL to call bokeh on.
4. Bokeh receives this request and creates a new Bokeh session. This runs all the code in `experiment/main.py` but it does AFAIK not render the Bokeh. It generates and renders all plots and triggers a async callback which loads the data after the response has been sent.
5. Bokeh replies with some JSON to the Django app and control is transferred back to our Django.
6. Our Django app uses some monkey patched Bokeh internals to load and render our bokeh app template. The template is served up to the user.
7. The user's browser loads resources from the Bokeh app and connects to it using a websocket. The Bokeh app URL is set inside our Django app using the environment variable `BOKEH_ROOT`.

I set these variables on the host:
BOKEH_ROOT=https://bokeh.airpublic.local
BOKEH_NUM_PROCESSES=1
DJANGO_SETTINGS_MODULE=thingapi.settings

## URLs for local development

| Use case | URL |
|-|-|
| Access Bokeh app directly | http://thingapi-internal.airpublic.local:5000/experiment?experiment_id=1 |
| Access Bokeh app directly | http://<thingapi_ip\>:5000/experiment?experiment_id=1 |
| Access Bokeh app directly | http://localhost:5000/experiment?experiment_id=1 |
| Access Django app directly | http://thingapi-internal.airpublic.local:8000/experiments/ |
| Access Django app directly | http://thingapi-internal.airpublic.local:8000/admin/ |
| Access Django app directly | http://<thingapi_ip\>:8000/experiments/ |
| Access Django app directly | http://<thingapi_ip\>:8000/admin/ |
| Access Django app directly | http://localhost:8000/experiments/ |
| Access Django app directly | http://localhost:8000/admin/ |
| Access Bokeh app through nginx | https://bokeh-0.bokeh.airpublic.local/experiment?experiment_id=1 |
| Access Django app through nginx | https://thingapi.airpublic.local/experiments/ |
| Access Django app through nginx | https://thingapi.airpublic.local/admin/ |

For some of these to work you might need to skip through your browser's security warnings since the certificates are self signed. You might have to do this for both Bokeh and Django for the dashboard to work locally

To find the IP address, you can do
```bash
docker network ls
docker network inspect a_result_of_previous_command # The IPs will be in the JSON
```

## Admin Usage

Visit http://localhost:8000/admin/

For the dashboard to work, we need to set up some metadata. You need one Experiment and at least one DeviceTimeRange associated with it.

A device time range specifies a time range (either end can be left open), a device, and a species. It also specifies a "Calibration model loader" and arguments for it. The fully qualified name of any python class can be specified as the calibration model loader. I recommend using `thingapi.loaders.HourlyDemo` to start with.

The calibration model loader is responsible for loading data, loading a calibration model, running it, and producing results for the specified time range, species, and device. But there's no validation yet and this is kind of deliberate so in practice it can return the raw data if we want it to without having to define every field in the database.

The fields in the DeviceTimeRange object should be used by the calibration model loader to determine what data to return. However, obeying them is not compulsary and therefore disobeying them will not cause any bugs. The dashboard must rely on the actual data instead of these fields in determining what options to give the user.

We also need a WeatherDataSource. This works in a similar way to DeviceTimeRange.

For an experiment to be visible to a given user, the `User` must be associated with a `Client`, and the `Experiment` must be associated with the `Client`.

## Dashboard development notebook

To start jupyter in the container:
```bash
docker exec -it thingapi jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

Visit http://localhost:8888 and select notebook.

The idea of the notebook is to import stuff from the bokeh app and override it for faster and more comfortable development. But it is currently slightly broken.


## Experiment config for Calibrated NO2
As loader use thingapi.loaders.HourlyV4Calibrated

Put the path to trained models into the arguments
```json
{"args": [], "kwargs": {"mobile": false, 
"no2_models_pickle": "thingapi/calibration/models_dicts.model"}}
```
