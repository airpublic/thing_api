import hashlib
import json
import time
import traceback
import binascii
import os
from datetime import date
from datetime import datetime
from datetime import timedelta
from importlib import import_module, reload
import functools
import methodtools
import dateutil.parser
import numpy as np
import pandas as pd
import pytz
from importlib import import_module, reload
from django.contrib.auth.models import User
import urllib3
from django.conf import settings
from django.db import models, transaction, connection
from django.db.models import Max
from django.db.models.fields import FloatField, IntegerField
from concurrent.futures import ThreadPoolExecutor


def no_loader_args():
    return json.dumps({"args": [], "kwargs": {}})


class Reading(models.Model):

    altitude         = models.FloatField(null=True)
    co               = models.FloatField(null=True)
    co_aux           = models.FloatField(null=True)
    co_working       = models.FloatField(null=True)
    gas_temp         = models.FloatField(null=True)
    gps_fix          = models.IntegerField(null=True)
    heading          = models.FloatField(null=True)
    humidity         = models.FloatField(null=True)
    latitude         = models.FloatField(null=True)
    lonet_bat        = models.IntegerField(null=True)
    longitude        = models.FloatField(null=True)
    no2              = models.FloatField(null=True)
    no2_aux          = models.FloatField(null=True)
    no2_working      = models.FloatField(null=True)
    pm1              = models.FloatField(null=True)
    pm10             = models.FloatField(null=True)
    pm2_5            = models.FloatField(null=True)
    w_pm1            = models.FloatField(null=True)
    w_pm10           = models.FloatField(null=True)
    w_pm2_5          = models.FloatField(null=True)
    sample_flow_rate = models.FloatField(null=True)
    sampling_period  = models.FloatField(null=True)
    speed            = models.FloatField(null=True)
    temp             = models.FloatField(null=True)
    date             = models.FloatField()
    retries          = models.IntegerField()
    last_organicity_sync = models.DateTimeField(null=True, blank=True)
    timestamp        = models.DateTimeField(default=datetime.now, primary_key=True)
    id               = models.IntegerField()

    def to_organicity_asset(self, experimenter_id, experiment_id):

        asset = {}

        organicity_id_hash = hashlib.md5()
        organicity_id_hash.update("{}_{}".format(self.id, self.timestamp).encode('utf-8'))
        organicity_id = organicity_id_hash.hexdigest()

        asset['id'] = "urn:oc:entity:experimenters:{}:{}:{}".format(experimenter_id, experiment_id, organicity_id)
        asset['type'] = "urn:oc:entityType:London:airQuality:airpublic:testing:reading"

        ts = self.timestamp.strftime('%Y-%m-%dT%H:%m:%S') + ".000Z"

        asset["TimeInstant"] = {"type": "urn:oc:attributeType:ISO8601", "value": ts}

        mapping = {
                    "altitude": "position:altitude",
                    "latitude": "position:latitude",
                    "longitude": "position:longitude",
                    "humidity": "relativeHumidity",
                    "temp": "temperature:ambient",
                    "co": "chemicalAgentAtmosphericConcentration:CO",
                    "no2": "chemicalAgentAtmosphericConcentration:NO2",
                    "pm1": "chemicalAgentAtmosphericConcentration:PM1",
                    "pm2_5": "chemicalAgentAtmosphericConcentration:PM2.5",
                    "pm10": "chemicalAgentAtmosphericConcentration:PM10",
                  }


        attributes = []

        for airpublic_key, organicity_key in mapping.items():

            organicity_type = "urn:oc:attributeType:{}".format(organicity_key)

            attribute = {
            #    "name": organicity_key,
                "type": organicity_type,
                "value": self.__getattribute__(airpublic_key),
#                "metadata": {}
            }

#            attributes.append(attribute)
            asset[airpublic_key] = attribute
#            #attributes[organicity_key] = attribute

#        attributes.append["TimeInstant"] = {"type": "urn:oc:attributeType:ISO8601", "value": ts}

#        asset['attributes'] = attributes

        return asset

    def is_worth_sending_to_organicity(self):

        return self.gps_fix == 1

    def organicity_sync_nonpersistent(self):

        experimenter_id = settings.ORGANICITY_SETTINGS['EXPERIMENTER_ID']
        experiment_id = settings.ORGANICITY_SETTINGS['EXPERIMENT_ID']
        application_id = settings.ORGANICITY_SETTINGS['APPLICATION_ID']
        client_id = settings.ORGANICITY_SETTINGS['CLIENT_ID']
        client_secret = settings.ORGANICITY_SETTINGS['CLIENT_SECRET']

        organicity = Organicity(client_id, client_secret)

        token = OrganicityToken.get_access_token(client_id, client_secret)
        asset = self.to_organicity_asset(experimenter_id, experiment_id)
        organicity.publish_asset(asset, token, application_id, experiment_id)

        self.last_organicity_sync = datetime.now()

    def organicity_sync_persistent(self, *args, **kwargs):
        self.organicity_sync_nonpersistent()
        super(Reading, self).save(*args, **kwargs)


    def attempt_organicity_sync(self):

        try:
            self.organicity_sync_nonpersistent()
            return None

        except OrganicityError as e:

            # Never mind
            return e

    def save(self, *args, **kwargs):

        if self.is_worth_sending_to_organicity():

            self.attempt_organicity_sync()

        return super(Reading, self).save(*args, **kwargs)


    class Meta:
        unique_together = (('id', 'timestamp'),)

def generate_api_key():
    return binascii.b2a_hex(os.urandom(32)).decode('UTF-8')

#class Device(models.Model):
#
#    id = models.IntegerField(primary_key=True)
#    api_key = models.CharField(max_length=64, default=generate_api_key)

class OrganicityError(Exception):
#    def __init__(e):
#        self.error = e

    pass

class Organicity():

    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret

    def get_token(self):

        http = urllib3.PoolManager()
        headers = urllib3.util.make_headers(basic_auth='{}:{}'.format(self.client_id, self.client_secret))
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        token_time_lb = time.time()
        try:

            r = http.request('POST', 'accounts.organicity.eu/realms/organicity/protocol/openid-connect/token', headers=headers, body="grant_type=client_credentials")

        except (urllib3.exceptions.HTTPError, urllib3.exceptions.PoolError, urllib3.exceptions.TimeoutError, urllib3.exceptions.ProtocolError, urllib3.exceptions.HTTPWarning) as e:
            raise OrganicityError(e)

        if r.status != 200:
            raise OrganicityError("Organicity returned status {} while getting token. {}".format(r.status, r.data))

        h = json.loads(r.data.decode('utf-8'))

        access_token_expires = token_time_lb + h['expires_in']
        refresh_token_expires = token_time_lb + h['refresh_expires_in']

        token = OrganicityToken.objects.create(access_token = h['access_token'], refresh_token = h['refresh_token'], access_token_expires = access_token_expires, refresh_token_expires = refresh_token_expires)
        return token

    def publish_asset(self, asset, token, application_id, experiment_id):

        headers = {}

        headers['X-Organicity-Application'] = application_id
        headers['X-Organicity-Experiment'] = experiment_id
        headers['Content-Type'] = 'application/json'
        headers['Accept'] = 'application/json'
        headers['Authorization'] = 'Bearer {}'.format(token.access_token)

        http = urllib3.PoolManager()

        try:

            try:
                req_json = json.dumps(asset, allow_nan=False)
            except ValueError:
                raise OrganicityError("Refusing to serialize NaN to JSON")

            r = http.request('POST', 'https://exp.orion.organicity.eu/v2/entities', headers=headers, body=req_json)

        except urllib3.exceptions.HTTPError as e:
            raise OrganicityError(e)
        except urllib3.exceptions.PoolError as e:
            raise OrganicityError(e)
        except urllib3.exceptions.TimeoutError as e:
            raise OrganicityError(e)
        except urllib3.exceptions.ProtocolError as e:
            raise OrganicityError(e)
        except urllib3.exceptions.HTTPWarning as e:
            raise OrganicityError(e)


        if r.status == 422 and json.loads(r.data.decode('utf-8')).get('description', None) == 'Already Exists':
            self.update_asset(asset, token, application_id, experiment_id)
        elif r.status != 201:
            raise OrganicityError("Organicity returned status {} during asset creation: {}".format(r.status, r.data))

    def delete_asset(self, asset, token, application_id, experiment_id):

        headers = {}

        headers['X-Organicity-Application'] = application_id
        headers['X-Organicity-Experiment'] = experiment_id
        headers['Accept'] = 'application/json'
        headers['Authorization'] = 'Bearer {}'.format(token.access_token)

        asset_id = asset['id']

        http = urllib3.PoolManager()

        try:

            url = 'https://exp.orion.organicity.eu/v2/entities/{}'.format(asset_id)

            r = http.request('DELETE', url, headers=headers, body=json.dumps(asset))

        except urllib3.exceptions.HTTPError as e:
            raise OrganicityError(e)
        except urllib3.exceptions.PoolError as e:
            raise OrganicityError(e)
        except urllib3.exceptions.TimeoutError as e:
            raise OrganicityError(e)
        except urllib3.exceptions.ProtocolError as e:
            raise OrganicityError(e)
        except urllib3.exceptions.HTTPWarning as e:
            raise OrganicityError(e)

        if r.status != 204:
            raise OrganicityError("Organicity returned status {} during asset update: {}".format(r.status, r.data))

    def update_asset(self, asset, token, application_id, experiment_id):

        headers = {}

        headers['X-Organicity-Application'] = application_id
        headers['X-Organicity-Experiment'] = experiment_id
        headers['Content-Type'] = 'application/json'
        headers['Accept'] = 'application/json'
        headers['Authorization'] = 'Bearer {}'.format(token.access_token)

        asset = asset.copy()
        asset_id = asset.pop('id')
        asset.pop('type')

        http = urllib3.PoolManager()

        try:

            url = 'https://exp.orion.organicity.eu/v2/entities/{}/attrs'.format(asset_id)

            r = http.request('POST', url, headers=headers, body=json.dumps(asset))

        except urllib3.exceptions.HTTPError as e:
            raise OrganicityError(e)
        except urllib3.exceptions.PoolError as e:
            raise OrganicityError(e)
        except urllib3.exceptions.TimeoutError as e:
            raise OrganicityError(e)
        except urllib3.exceptions.ProtocolError as e:
            raise OrganicityError(e)
        except urllib3.exceptions.HTTPWarning as e:
            raise OrganicityError(e)

        if r.status != 204:
            raise OrganicityError("Organicity returned status {} during asset update: {}".format(r.status, r.data))


class OrganicityToken(models.Model):

    access_token = models.CharField(max_length=2048)
    refresh_token = models.CharField(max_length=2048)
    access_token_expires = models.IntegerField()
    refresh_token_expires = models.IntegerField()
    being_refreshed = models.IntegerField(default=False)

    @classmethod
    def expiry_time(klass, time_good_for=10):

        db_roundtrip_max_time = 2

        return int(np.ceil(time.time())) + time_good_for + db_roundtrip_max_time

    @classmethod
    def get_existing_access_token(klass, time_good_for=10):

        return klass.objects.filter(access_token_expires__gte = klass.expiry_time(time_good_for)).first()

    @classmethod
    def get_existing_refresh_token(klass, time_good_for=10):

        return klass.objects.filter(refresh_token_expires__gte = klass.expiry_time(time_good_for)).first()

    @classmethod
    def cleanup_expired(klass):

        now_lenient = int(np.ceil(time.time())) - 10

        klass.objects.filter(access_token_expires__lte = now_lenient).filter(refresh_token_expires__lte = now_lenient).delete()


    @classmethod
    @transaction.atomic
    def get_access_token(klass, client_id, client_secret, time_good_for=10):

        # This is annoying and slow, but it avoids the need for a background task.

        # First, try to find a valid access token with 10s left
        #   Use access token
        # If not found, find a refresh token with 10s left
        #   Refresh access token
        #   Use access token
        # If not found, use the service account to get a new access token.

        # Cleanup step by deleting all nearly-expired access tokens

        access_token = klass.get_existing_access_token(time_good_for)

        if access_token:
            return access_token
        else:
            # There's no point in refreshing unless we do it periodically in the background, since we can just request a new token
            #refresh_token = klass.get_existing_refresh_token(time_good_for)

            return Organicity(client_id, client_secret).get_token()


class RawReading(models.Model):

    altitude         = models.FloatField(null=True)
    m_co             = models.FloatField(null=True)
    m_no2            = models.FloatField(null=True)
    co_a             = models.FloatField(null=True)
    co_w             = models.FloatField(null=True)
    pt               = models.FloatField(null=True)
    gps_fix          = models.IntegerField(null=True)
    heading          = models.FloatField(null=True)
    humidity         = models.FloatField(null=True)
    latitude         = models.FloatField(null=True)
    lonet_bat        = models.IntegerField(null=True)
    longitude        = models.FloatField(null=True)
    no2_a            = models.FloatField(null=True)
    no2_w            = models.FloatField(null=True)
    pm1              = models.FloatField(null=True)
    pm10             = models.FloatField(null=True)
    pm2_5            = models.FloatField(null=True)
    w_pm1            = models.FloatField(null=True)
    w_pm10           = models.FloatField(null=True)
    w_pm2_5          = models.FloatField(null=True)
    sample_flow_rate = models.FloatField(null=True)
    sampling_period  = models.FloatField(null=True)
    speed            = models.FloatField(null=True)
    temp             = models.FloatField(null=True)
    bat_v            = models.FloatField(null=True)
    location_std     = models.FloatField(null=True)
    date             = models.FloatField()
    retries          = models.IntegerField()
    last_organicity_sync = models.DateTimeField(null=True, blank=True)
    timestamp        = models.DateTimeField(default=datetime.now, primary_key=True)
    id               = models.IntegerField()

    class Meta:
        unique_together = (('id', 'timestamp'),)

    @classmethod
    def dummy(self):
        return RawReading(**{
            a.name: 0
            for a in RawReading._meta.fields
            if isinstance(a, (FloatField, IntegerField))
        })


class RawReadingV4(models.Model):
    ver = models.IntegerField(null=True)
    rtys = models.IntegerField(null=True)
    vol = models.FloatField(null=True)
    afewrk1 = models.FloatField(null=True)
    afeaux1 = models.FloatField(null=True)
    afewrk2 = models.FloatField(null=True)
    afeaux2 = models.FloatField(null=True)
    afewrk3 = models.FloatField(null=True)
    afeaux3 = models.FloatField(null=True)
    afept1k = models.FloatField(null=True)
    isbwrk = models.FloatField(null=True)
    isbaux = models.FloatField(null=True)
    mics1 = models.FloatField(null=True)
    mics2 = models.FloatField(null=True)
    time = models.CharField(null=True, max_length=256, blank=True)
    lat = models.FloatField(null=True)
    long = models.FloatField(null=True)
    alt = models.FloatField(null=True)
    fix = models.IntegerField(null=True)
    speed = models.FloatField(null=True)
    head = models.FloatField(null=True)
    rpm = models.FloatField(null=True)
    pm11c = models.FloatField(null=True)
    pm125c = models.FloatField(null=True)
    pm110c = models.FloatField(null=True)
    pm11a = models.FloatField(null=True)
    pm125a = models.FloatField(null=True)
    pm110a = models.FloatField(null=True)
    pm1par3 = models.FloatField(null=True)
    pm1par5 = models.FloatField(null=True)
    pm1par10 = models.FloatField(null=True)
    pm1par25 = models.FloatField(null=True)
    pm1tmp = models.FloatField(null=True)
    pm1hum = models.FloatField(null=True)
    pm21c = models.FloatField(null=True)
    pm225c = models.FloatField(null=True)
    pm210c = models.FloatField(null=True)
    pm21a = models.FloatField(null=True)
    pm225a = models.FloatField(null=True)
    pm210a = models.FloatField(null=True)
    pm2par3 = models.FloatField(null=True)
    pm2par5 = models.FloatField(null=True)
    pm2par10 = models.FloatField(null=True)
    pm2par25 = models.FloatField(null=True)
    pm2tmp = models.FloatField(null=True)
    pm2hum = models.FloatField(null=True)
    timestamp        = models.DateTimeField(default=datetime.now, primary_key=True)
    id               = models.IntegerField()

    class Meta:
        unique_together = (('id', 'timestamp'),)


class LaqnGroup(models.Model):

    group_name = models.CharField(max_length=2048, null=False, blank=False, primary_key=True)
    description = models.CharField(max_length=2048, null=False, blank=True)
    website_url = models.CharField(max_length=2048, null=False, blank=True)
    last_laqn_sync = models.DateTimeField(default=datetime.now)

class LaqnSite(models.Model):

    site_code = models.CharField(max_length=2048, null=False, blank=False, primary_key=True)
    local_authority_code = models.CharField(max_length=2048, null=False, blank=False)
    local_authority_name = models.CharField(max_length=2048, null=False, blank=False)
    site_name = models.CharField(max_length=2048, null=False, blank=False)
    site_type = models.CharField(max_length=2048, null=False, blank=False)
    date_closed = models.DateTimeField(null=True, blank=True)
    date_opened = models.DateTimeField(null=True, blank=True)
    latitude = models.FloatField(null=True)
    longitude = models.FloatField(null=True)
    latitude_wgs84 = models.FloatField(null=True)
    longitude_wgs84 = models.FloatField(null=True)
    data_owner = models.CharField(max_length=2048, null=False, blank=True)
    data_manager = models.CharField(max_length=2048, null=False, blank=True)
    site_link = models.CharField(max_length=2048, null=False, blank=True)
    last_laqn_sync = models.DateTimeField(default=datetime.now)

class LaqnReading(models.Model):

    site_code = models.CharField(max_length=2048, null=False)
    species_code = models.CharField(max_length=2048, null=False)
    timestamp = models.DateTimeField(null=True, blank=False)
    value = models.FloatField(null=True)
    last_laqn_sync = models.DateTimeField(default=datetime.now)

    class Meta:
        unique_together = (('site_code', 'timestamp', 'species_code'),)


def import_laqn_groups():

    http = urllib3.PoolManager()

    r = http.request('GET', 'api.erg.kcl.ac.uk/AirQuality/Information/Groups/Json')

    groups_resp = json.loads(r.data.decode('utf-8'))

    groups_resp_groups = groups_resp.get("Groups")
    groups = groups_resp_groups.get("Group")

    cur = connection.cursor()

    placeholders = ",".join(["(%s, %s, %s, NOW())" for group in groups])

    values = [item for group in groups for item in [group["@GroupName"],
                                                    group["@Description"],
                                                    group["@WebsiteURL"]]]

    group_table_name = LaqnGroup._meta.db_table

    query_string = """
INSERT INTO {} (group_name, description, website_url, last_laqn_sync)
VALUES {}
ON CONFLICT(group_name) DO
UPDATE SET group_name = EXCLUDED.group_name,
           description = EXCLUDED.description,
           website_url = EXCLUDED.website_url
""".format(group_table_name,
           placeholders)

    cur.execute(query_string, values)

def laqn_parse_date(datestr):
    if datestr is "":
        return None
    else:
        dt = dateutil.parser.parse(datestr)
        return pytz.timezone("GMT").localize(dt)

def laqn_parse_float(floatstr):
    if floatstr is "" or floatstr is None:
        return None
    try:
        return float(floatstr)
    except ValueError as e:
        print("Ha! Found the problematic float! It was {}".format(floatstr))
        raise e

def import_laqn_sites(group_name):

    http = urllib3.PoolManager()

    r = http.request('GET', 'api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName={}/Json'.format(group_name))

    sites_resp = json.loads(r.data.decode('utf-8'))

    sites_resp_sites = sites_resp.get("Sites")
    sites = sites_resp_sites.get("Site")

    cur = connection.cursor()

    placeholders = ",".join([
        "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())"
        for site in sites])

    values = [item for site in sites
                   for item in [site["@LocalAuthorityCode"],
                                site["@LocalAuthorityName"],
                                site["@SiteCode"],
                                site["@SiteName"],
                                site["@SiteType"],
                                laqn_parse_date(site["@DateClosed"]),
                                laqn_parse_date(site["@DateOpened"]),
                                laqn_parse_float(site["@Latitude"]),
                                laqn_parse_float(site["@Longitude"]),
                                laqn_parse_float(site["@LatitudeWGS84"]),
                                laqn_parse_float(site["@LongitudeWGS84"]),
                                site["@DataOwner"],
                                site["@DataManager"],
                                site["@SiteLink"]]]

    site_table_name = LaqnSite._meta.db_table

    query_string = """
INSERT INTO {} (local_authority_code,
                local_authority_name,
                site_code,
                site_name,
                site_type,
                date_closed,
                date_opened,
                latitude,
                longitude,
                latitude_wgs84,
                longitude_wgs84,
                data_owner,
                data_manager,
                site_link,
                last_laqn_sync)
VALUES {}
ON CONFLICT(site_code) DO
UPDATE SET local_authority_code = EXCLUDED.local_authority_code,
           local_authority_name = EXCLUDED.local_authority_name,
           site_name = EXCLUDED.site_name,
           site_type = EXCLUDED.site_type,
           date_closed = EXCLUDED.date_closed,
           date_opened = EXCLUDED.date_opened,
           latitude = EXCLUDED.latitude,
           longitude = EXCLUDED.longitude,
           latitude_wgs84 = EXCLUDED.latitude_wgs84,
           longitude_wgs84 = EXCLUDED.longitude_wgs84,
           data_owner = EXCLUDED.data_owner,
           data_manager = EXCLUDED.data_manager,
           site_link = EXCLUDED.site_link
""".format(site_table_name,
           placeholders)

    cur.execute(query_string, values)


class LaqnKnownErrorPresumableCausedByNoDataAvailable(Exception):
    pass


def import_laqn_data(site_code, start_date, end_date):

    http = urllib3.PoolManager()

    r = http.request('GET', 'api.erg.kcl.ac.uk/AirQuality/Data/Site/SiteCode={}/StartDate={}/EndDate={}/Json'.format(site_code, start_date, end_date))

    r_utf8 = r.data.decode('utf-8')

    try:
        data_resp = json.loads(r_utf8)
    except json.JSONDecodeError as e:
        if 'There is no row at position 0' in r_utf8:
            print("Aborting due to a known issue in LAQN")
            raise LaqnKnownErrorPresumableCausedByNoDataAvailable()
        else:
            raise e

    api_aq_data = data_resp.get("AirQualityData")
    api_data = api_aq_data["Data"]

    # Dedupe data

    api_data = {(site_code,
                 reading["@SpeciesCode"],
                 laqn_parse_date(reading["@MeasurementDateGMT"]),
                 ): laqn_parse_float(reading["@Value"])
                for reading in api_data}

    cur = connection.cursor()

    placeholders = ",".join(["(%s, %s, %s, %s, NOW())" for group in api_data])

    values = [item for (st, sp, ts), v in api_data.items()
                   for item in [st, sp, ts, v]]

    data_table_name = LaqnReading._meta.db_table

    query_string = """
INSERT INTO {} (site_code, species_code, timestamp, value, last_laqn_sync)
VALUES {}
ON CONFLICT(site_code, species_code, timestamp) DO
UPDATE SET value = EXCLUDED.value
""".format(data_table_name,
           placeholders)

    cur.execute(query_string, values)


def site_laqn_update(site, today):
    def func():

        try:
            end_date = None

            def site_closed(end_date):
                return end_date is not None and site.date_closed is not None and site.date_closed.date() <= end_date

            while ((end_date is None or (end_date < today)) and not site_closed(end_date)):
                end_date = import_laqn_data_batch(site, today)
        except LaqnKnownErrorPresumableCausedByNoDataAvailable:
            return
        except Exception as e:
            traceback.print_exc(e)
            raise e

    return func


def import_laqn_update():

    import_laqn_groups()
    import_laqn_sites('All')



    for site in LaqnSite.objects.all():

        today = date.today()

        site_laqn_update(site, today)()

#    with ThreadPoolExecutor(max_workers=128) as pool:
#
#        today = date.today()
#
#        sites = LaqnSite.objects.all()
#
#        pool.map(lambda s: site_laqn_update(s, today)(), sites)
#        for site in LaqnSite.objects.all():
#
#            pool.submit(site_laqn_update(site, today))

def import_laqn_data_batch(site, today):
    readings = LaqnReading.objects.filter(site_code=site.site_code)
    latest_reading = readings.aggregate(Max('timestamp'))

    earliest_wanted_reading = datetime(2017, 2, 8, )
    earliest_wanted_reading = pytz.timezone("GMT").localize(earliest_wanted_reading)

    start_dt = latest_reading["timestamp__max"]

    if start_dt is None:
        start_dt = site.date_opened

    if start_dt < earliest_wanted_reading:
        start_dt = earliest_wanted_reading

    start_date = start_dt.date()

    end_date = (start_dt + timedelta(30, 0)).date()

    if site.date_closed is not None and end_date > site.date_closed.date():
        end_date = site.date_closed.date()

    if end_date > today + timedelta(1, 0):
        end_date = today + timedelta(1, 0)

    print("importing site {} from {} to {}".format(site.site_code,
                                                   start_date,
                                                   end_date))

    import_laqn_data(site.site_code, start_date, end_date)

    return end_date


def get_device_last_reading(id):

    assert isinstance(id, int), "id should be an integer, was %s" % type(id)

    return pd.read_sql_query("SELECT * FROM thingapi_rawreading WHERE id = %d ORDER BY timestamp DESC LIMIT 1" % id, connection)


def get_device_last_reading_v4(id):

    assert isinstance(id, int), "id should be an integer, was %s" % type(id)

    return pd.read_sql_query("SELECT * FROM thingapi_rawreadingv4 WHERE id = %d ORDER BY timestamp DESC LIMIT 1" % id, connection)


class NamedDjangoModel:
    def __str__(self):
        if self.id is None:
            return "New %s (%s)" % (type(self).__name__, self.name or "No name")
        else:
            return "%s #%d (%s)" % (type(self).__name__, self.id, self.name)


class Client(NamedDjangoModel, models.Model):
    name = models.CharField(max_length=256)
    experiments = models.ManyToManyField("Experiment",
                                         through="ClientExperiment",
                                         through_fields=("client", "experiment"))
    users = models.ManyToManyField(User,
                                   through="ClientUser",
                                   through_fields=("client", "user"),
                                   related_name="clients")


class Experiment(NamedDjangoModel, models.Model):
    name = models.CharField(max_length=256)
    clients = models.ManyToManyField(Client,
                                     through="ClientExperiment",
                                     through_fields=("experiment", "client"))

    @methodtools.lru_cache()
    def get_loaders(self):
        return [
            dtr.instantiate_model()
            for dtr in self.device_time_ranges.all()
        ]

    def load_data(self):
        executor = ThreadPoolExecutor(max_workers=30)

        # TODO - add timeout
        calibrated_results_list = [DeviceTimeRange.empty_result_df()]

        for df, loader in zip(
                executor.map(lambda loader: loader.results_as_dataframe(),
                             self.get_loaders()),
                self.get_loaders()):

            # If empty series are plotted, holoviews/Bokeh messes up the labels
#            if df.shape[0] == 0:
#                df = self.get_loader_placeholder_data(loader=loader)

            calibrated_results_list.append(df)

        return pd.concat(calibrated_results_list, sort=False)


class ClientExperiment(models.Model):
    client = models.ForeignKey(Client, null=False,
                               on_delete=models.CASCADE,
                               related_name="client_experiments")
    experiment = models.ForeignKey(Experiment, null=False,
                                   on_delete=models.CASCADE,
                                   related_name="client_experiments")


def instantiate_by_name_with_args(name, *args, **kwargs):

    module_path, klass_name = name.rsplit('.', 1)

    module = import_module(module_path)
    if settings.DEBUG:
        reload(module)
    klass_name = getattr(module, klass_name)

    return klass_name(*args, **kwargs)


class DeviceTimeRange(models.Model):
    """
    One device over one timerange but all species.
    Has the calibration model in here and it can be decided by calibration_model_args. for example, can serve raw data
    One experiment has many DeviceTimeRanges
    Supports multiple species in one dataframe

    Usual use is calibration_model_loader = 'thingapi.loaders.HourlyDemo'
    """

    start_timestamp = models.DateTimeField(null=True, blank=True)
    end_timestamp = models.DateTimeField(null=True, blank=True)
    experiment = models.ForeignKey(Experiment, null=False, on_delete=models.CASCADE,
                                   related_name="device_time_ranges")
    device_id = models.IntegerField(null=False)
    calibration_model_loader = models.CharField(max_length=128, null=False)
    calibration_model_args = models.TextField(null=False, default=no_loader_args)
    species = models.CharField(max_length=32, null=False, blank=False)


    @classmethod
    def empty_result_df(self):
        df = pd.DataFrame({
            "timestamp": [], "latitude": [], "longitude": [], "species": [],
            "value": [], "device_id": [], "unit": []})

        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize('UTC')
        df["species"] = df["species"].astype(str)
        df["device_id"] = df["species"].astype(str)
        return df

    def instantiate_model(self):
        arg_values = json.loads(self.calibration_model_args)

        return instantiate_by_name_with_args(
            self.calibration_model_loader,
            self, *arg_values["args"], **arg_values["kwargs"]
        )


    def validate(self, result):
        """
        This function is here to prevent obscure errors being caused by
        models spewing out malformed output.

        But for now anything goes

        Remove negative values, very high ones, nans in the species column.
        This allows calibration model to stay as modelled.
        """

        return result


class METSite(models.Model):
    site_id = models.IntegerField(null=False)
    mode = models.CharField(null=False, max_length=8)
    name = models.CharField(null=False, max_length=128)
    latitude = models.FloatField(null=False)
    longitude = models.FloatField(null=False)

    class Meta:
        unique_together = (('site_id', 'mode'),)


class METField(models.Model):
    name = models.CharField(null=False, blank=False, max_length=128)
    unit = models.CharField(null=False, blank=False, max_length=128)

    class Meta:
        unique_together = (('name', 'unit'),)


class METPeriod(models.Model):
    start_time = models.DateTimeField(null=False, blank=False)
    duration = models.IntegerField(null=False)

    class Meta:
        unique_together = (('start_time', 'duration'),)


class METReading(models.Model):
    period = models.ForeignKey(METPeriod, on_delete=models.CASCADE, null=False)
    field = models.ForeignKey(METField, on_delete=models.CASCADE, null=False)
    site = models.ForeignKey(METSite, on_delete=models.CASCADE, null=False)
    value = models.CharField(max_length=16)

    class Meta:
        unique_together = (('period', 'field', 'site'),)


def create_all_data_experiment_v3():
    exp = Experiment.objects.create(name="All data")

    device_ids = list(set([item[0] for item in RawReading.objects.values_list('id')]))

    for i in device_ids:
        DeviceTimeRange.objects.create(
            device_id=i, experiment_id=exp.id,
            calibration_model_loader='thingapi.loaders.HourlyV3'
        )

    return exp


def create_all_data_experiment_v4():
    exp = Experiment.objects.create(name="All data v4")

    device_ids = list(set([item[0] for item in RawReadingV4.objects.values_list('id')]))

    for i in device_ids:
        DeviceTimeRange.objects.create(
            device_id=i, experiment_id=exp.id,
            calibration_model_loader='thingapi.loaders.HourlyV4',
            calibration_model_args=json.dumps(
                {"args": [], "kwargs": {"mobile": False}}
            )
        )

    return exp


class WeatherDataSource(models.Model):
    name = models.CharField(null=False, blank=False,
                            unique=True, max_length=256)
    weather_loader = models.CharField(max_length=128, null=False)
    weather_args = models.TextField(null=False, default=no_loader_args)

    def instantiate_loader(self):
        arg_values = json.loads(self.weather_args)

        return instantiate_by_name_with_args(
            self.weather_loader,
            *arg_values["args"], **arg_values["kwargs"]
        )


class ClientUser(models.Model):

    user = models.ForeignKey(User, null=False, on_delete=models.CASCADE,
                             related_name="client_users")
    client = models.ForeignKey(Client, null=False, on_delete=models.CASCADE,
                               related_name="client_users")

    class Meta:
        unique_together = ("user", "client")


class PrecomputedLoaderData(models.Model):
    device_time_range = models.ForeignKey("DeviceTimeRange",
            null=False, on_delete=models.CASCADE)
    version = models.PositiveIntegerField(default=1)
    results_as_dataframe = models.BinaryField()
    device_ids_as_json = models.CharField(max_length=1024)
    species_as_json = models.CharField(max_length=1024)
    dummy_locations = models.BinaryField()
    updated_at = models.DateTimeField(default=datetime.now)


class ApiKey(models.Model):
    created = models.DateTimeField(default=datetime.now)
    last_used = models.DateTimeField(default=None, null=True)
    key = models.CharField(null=False, blank=False, unique=False, max_length=128, default=generate_api_key)
    client = models.ForeignKey(Client, null=False, on_delete=models.CASCADE, related_name="api_keys")
    user_created = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)


class Alert(models.Model):
    created = models.DateTimeField(default=datetime.now)
    recipients_json = models.TextField(null=False, blank=False, default="[]")
    experiment = models.ForeignKey(Experiment, null=False, on_delete=models.CASCADE, related_name='alerts')
    condition_class = models.CharField(null=False, blank=False, max_length=256)
    condition_args = models.TextField(null=False, blank=False, default=no_loader_args())
    last_changed = models.DateTimeField(default=None, blank=True, null=True)
    status = models.BooleanField(null=False, default=False)

    @classmethod
    def poll_all(self, current_time=None):

        alerts = Alert.objects.all()

        alerts_by_experiment = {}

        for a in alerts:
            alert_list = alerts_by_experiment.get(a.experiment_id, [])
            alert_list.append(a)
            alerts_by_experiment[a.experiment_id] = alert_list

        for experiment_id, alerts in alerts_by_experiment.items():
            experiment = Experiment.objects.filter(id=experiment_id).first()

            changed_alerts = []

            data = experiment.load_data()

            for alert in alerts:
                condition = alert.instantiate_condition()

                new_status = condition.should_alert(data)
                if new_status != current_status:
                    changed_alerts.append({
                        'current_status': alert.current_status,
                        'new_status': new_status,
                        'alert': alert
                    })

                    alert.current_status = new_status
                    alert.save()

            del data

    def instantiate_condition(self):
        arg_values = json.loads(self.condition_args)

        return instantiate_by_name_with_args(
            self.condition_class,
            self,
            *arg_values["args"],
            **arg_values["kwargs"]
        )


class RawLoraReadingV4(models.Model):
    id = models.IntegerField()
    timestamp = models.DateTimeField(primary_key=True)
    ver = models.IntegerField()
    rtys = models.IntegerField()
    vol = models.FloatField()
    afewrk1 = models.FloatField()
    afeaux1 = models.FloatField()
    afewrk2 = models.FloatField()
    afeaux2 = models.FloatField()
    afewrk3 = models.FloatField()
    afeaux3 = models.FloatField()
    afept1k = models.FloatField()
    isbwrk = models.FloatField()
    isbaux = models.FloatField()
    mics1 = models.FloatField()
    mics2 = models.FloatField()
    rpm = models.FloatField()
    lat = models.FloatField()
    long = models.FloatField()
    pm11c = models.FloatField()
    pm125c = models.FloatField()
    pm110c = models.FloatField()
    pm1tmp = models.FloatField()
    pm1hum = models.FloatField()
    pm21c = models.FloatField()
    pm225c = models.FloatField()
    pm210c = models.FloatField()
    pm2tmp = models.FloatField()
    pm2hum = models.FloatField()

    class Meta:
        unique_together = (('id', 'timestamp'),)
