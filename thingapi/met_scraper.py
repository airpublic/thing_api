from thingapi.models import METSite, METField, METPeriod, METReading
from psycopg2.extensions import adapt
from django.db import connection, transaction
from datetime import timedelta, datetime
from dateutil.parser import isoparse
import requests
import pytz
import geopandas as gpd
from shapely.geometry.point import Point
import pyproj
import fiona.crs


class METAPIHelper:

    def met_request(self, product, path, **kwargs):
        url = 'http://datapoint.metoffice.gov.uk/public/data/val/%s/all/%s/%s' % (
            product, 'json', path
        )

        params = kwargs.copy()
        params['key'] = settings.MET_API_KEY

        resp = requests.get(url, params)

        if resp.status_code == 200:
            return resp.json()
        else:
            raise ValueError("MET office returned an error")

    def all_hourly_sites(self):
        return self.met_request('wxobs', 'sitelist')['Locations']['Location']

    def hourly_historic_24h_data(self, site_id):
        return self.met_request('wxobs', site_id, res='hourly')['SiteRep']


class METConvertor:

    def extract_rep(self, item):
        if type(item["Rep"]) is dict:
            return [item["Rep"]]
        else:
            return item["Rep"]

    def convert_sites(self, sites):
        site_data = [{
            "site_id": site["id"],
            "mode": "wxobs",
            "name": site["name"],
            "latitude": float(site["latitude"]),
            "longitude": float(site["longitude"])
        } for site in sites ]

        return site_data

    def convert_fields(self, data):
        meta = data['Wx']['Param']
        fields = [{
            "name": meta_item['$'],
            "unit": meta_item["units"]
        } for meta_item in meta]
        return fields

    def parse_date(self, datestr):
        """
        No libraries can handle this format: "2018-09-01Z"
        """
        if datestr[-1] == 'Z':
            return isoparse(datestr[:-1]).replace(tzinfo=pytz.utc)
        else:
            raise ValueError("Date format changed")

    def reconstitute_datetime(self, datestr, minutes):
        return self.parse_date(datestr) + timedelta(0, 60*int(minutes))

    def convert_periods(self, data):
        return [
            {
                "start_time": self.reconstitute_datetime(
                    day_item['value'], item['$']),
                "duration": 3600
            }
            for day_item in data['DV'].get('Location', {}).get('Period', [])
            for item in self.extract_rep(day_item)
        ]

    def convert_values(self, data, site, met_fields, met_periods):
        meta = data['Wx']['Param']
        met_fields_lookup = {
            field.name: field.id
            for field in met_fields
        }
        field_name_id_lookup = {
            meta_item['name']: met_fields_lookup[meta_item['$']]
            for meta_item in meta
        }

        period_id_lookup = {
            (period.start_time.isoformat(), period.duration): period.id
            for period in met_periods
        }

        periods_readings = [
            (
                {
                    "start_time": self.reconstitute_datetime(
                        day_item['value'], item['$']),
                    "duration": 3600
                },
                [
                    {
                        "field_id": field_name_id_lookup[k],
                        "value": v
                    }
                    for k, v in item.items()
                    if k != '$'
                ]
            )
            for day_item in data['DV'].get('Location', {}).get('Period', [])
            for item in self.extract_rep(day_item)
        ]

        readings = [
            {
                "site_id": site.id,
                "period_id": period_id_lookup[(period['start_time'].isoformat(),
                                               period['duration'])],
                "field_id": field["field_id"],
                "value": field["value"]
            }
            for period, fields in periods_readings
            for field in fields
        ]

        return readings


class METInsertor:

    def insert_sites(self, site_data):
        cur = connection.cursor()

        values_list = ",".join(["(%d, %s, %s, %f, %f)" % (
            int(site['site_id']),
            adapt(site['mode']),
            adapt(site['name']),
            site['latitude'],
            site['longitude']
        ) for site in site_data])

        tn = METSite._meta.db_table

        cur.execute("""
        INSERT INTO %s (site_id, mode, name, latitude, longitude) VALUES %s
        ON CONFLICT (site_id, mode) DO
        UPDATE SET name = EXCLUDED.name,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude
        """ % (tn, values_list))

        sites = METSite.objects.raw("""
        SELECT %s.*
        FROM %s
        INNER JOIN (VALUES %s) data (site_id, mode, a, b, c)
                ON data.site_id = %s.site_id
               AND data.mode = %s.mode
        """ % (tn, tn, values_list, tn, tn))

        return sites

    def insert_fields(self, fields_data):
        cur = connection.cursor()

        values_placeholders = ",".join([
            "(%s, %s)"
            for field_data in fields_data
        ])

        quoted_values_placeholders = ",".join([
            "('%s', '%s')"
            for field_data in fields_data
        ])

        def values():
            for field_data in fields_data:
                yield field_data['name']
                yield field_data['unit']

        def quoted_values():
            for val in values():
                yield adapt(val)

        tn = METField._meta.db_table

        cur.execute(("""
        INSERT INTO %s (name, unit) VALUES %s
        ON CONFLICT (name, unit) DO NOTHING
        """ % (tn, values_placeholders)) % tuple(quoted_values()))

        fields = METField.objects.raw("""
        SELECT %s.*
        FROM %s
        INNER JOIN (VALUES %s) data (name, unit)
                ON data.name = %s.name
               AND data.unit = %s.unit
        """ % (tn, tn, values_placeholders, tn, tn), list(values()))

        return list(fields)

    def insert_periods(self, periods_data):
        cur = connection.cursor()

        values_placeholders = ",".join([
            "(%s, %s)"
            for period_data in periods_data
        ])

        quoted_values_placeholders = ",".join([
            "('%s', '%s')"
            for period_data in periods_data
        ])

        def values():
            for period_data in periods_data:
                yield period_data['start_time']
                yield period_data['duration']

        def quoted_values():
            for val in values():
                yield adapt(val)

        tn = METPeriod._meta.db_table

        cur.execute(("""
        INSERT INTO %s (start_time, duration) VALUES %s
        ON CONFLICT (start_time, duration) DO NOTHING
        """ % (tn, values_placeholders)) % tuple(quoted_values()))

        periods = METPeriod.objects.raw("""
        SELECT %s.*
        FROM %s
        INNER JOIN (VALUES %s) data (start_time, duration)
                ON data.start_time = %s.start_time
               AND data.duration = %s.duration
        """ % (tn, tn, values_placeholders, tn, tn), list(values()))

        return list(periods)

    def insert_values(self, readings_data):
        cur = connection.cursor()

        readings_placeholders = ",".join([
            "(%s, %s, %s, %s)"
            for _ in readings_data
        ])

        def quoted_readings():
            for reading_data in readings_data:
                yield adapt(reading_data['site_id'])
                yield adapt(reading_data['period_id'])
                yield adapt(reading_data['field_id'])
                yield adapt(reading_data['value'])

        tn = METReading._meta.db_table

        cur.execute(("""
        INSERT INTO %s (site_id, period_id, field_id, value) VALUES %s
        ON CONFLICT (site_id, period_id, field_id) DO NOTHING
        """ % (tn, readings_placeholders)) % tuple(quoted_readings()))


def refresh_met_sites():

    wgs84 = pyproj.Proj(proj='latlong',datum='WGS84')
    croydon = Point(-0.0987762, 51.3763008)
    croydon_centric = fiona.crs.from_string('+proj=aeqd +lon0={} +lat0={} +units=m'.format(croydon.x, croydon.y))
    croydon_croydon_centric = pyproj.transform(wgs84, pyproj.Proj(**croydon_centric), croydon.x, croydon.y)

    api = METAPIHelper()
    conv = METConvertor()
    ins = METInsertor()

    hourly_sites = api.all_hourly_sites()

    hourly_site_df = gpd.GeoDataFrame({
        "geometry": [Point(float(s['longitude']), float(s['latitude'])) for s in hourly_sites],
        "id": [site['id'] for site in hourly_sites],
        "name": [site['name'] for site in hourly_sites]
    }, crs=wgs84.srs)

    hourly_site_df["distance_to_croydon"] = hourly_site_df.geometry.to_crs(croydon_centric).distance(Point(croydon_croydon_centric))

    close_hourly_site_ids = hourly_site_df[hourly_site_df["distance_to_croydon"] < 4e4]

    site_data = conv.convert_sites(hourly_sites)
    sites_obj = ins.insert_sites(site_data)

    for site_id in list(close_hourly_site_ids.id):
        site = METSite.objects.filter(site_id=site_id).first()
        weather_data = api.hourly_historic_24h_data(site_id)
        fields = conv.convert_fields(weather_data)
        if fields:
            fields_objs = ins.insert_fields(fields)
        periods = conv.convert_periods(weather_data)
        if periods:
            periods_objs = ins.insert_periods(periods)

        if fields and periods:
            values = conv.convert_values(weather_data, site, fields_objs, periods_objs)
            values_objs = ins.insert_values(values)

    return hourly_site_df
