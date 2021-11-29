FROM python:3.7-buster

RUN apt -y update --fix-missing

#RUN curl -sL https://deb.nodesource.com/setup_8.x | bash -
RUN apt -y install libproj-dev libgeos-dev postgresql-client libgdal-dev nodejs cron vim git htop nmap nethogs

# To get bokeh to build on my machine I used nvm to install node 8.8.1 and npm 6.4.1
#RUN npm install -g npm@6.4.1

# These are required for Cartopy to be installed
# Putting them in requirements.txt doesn't work
RUN pip install "Cython==0.29.24"
RUN pip install "numpy==1.16.6"

## Build bokeh. This should match the version in requirements.txt
## We do this here because of the --build-js option
#RUN git clone --depth 1 --branch 1.0.1 https://github.com/bokeh/bokeh /bokeh
#WORKDIR /bokeh
#
## Bokeh dev deps
#RUN pip install jinja2 pyyaml numpy requests tornado python-dateutil && python -m scripts.deps
#
#WORKDIR /bokeh/bokehjs
#RUN npm install --no-save
#
#WORKDIR /bokeh
#RUN python setup.py install --build-js

#RUN pip install "Proj4==4.9.0"
# Other deps

ADD . /thingapi
WORKDIR /thingapi
RUN pip install --upgrade pip
RUN pip install 'Cartopy==0.17.0' 'Shapely==1.7.0' --no-binary Cartopy --no-binary Shapely --no-build-isolation
RUN pip install -r requirements.txt

RUN cat crontab >> /etc/crontab


ENTRYPOINT /thingapi/docker-entrypoint.sh
