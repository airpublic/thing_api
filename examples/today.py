import holoviews as hv
from holoviews.streams import Buffer, Stream
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import reduce
from operator import mul
import streamz
import streamz.dataframe


import os
os.environ['DJANGO_SETTINGS_MODULE'] = os.environ.get('DJANGO_SETTINGS_MODULE', 'thingapi.settings')
import django
django.setup()


from django.db import connection


# Calibration params from leicester data
calib = pd.read_csv('data/linear_model_leicester_only.csv', index_col=0)

renderer = hv.renderer('bokeh').instance(mode='server')

start_time = datetime.utcnow() - timedelta(2, 0)


def retry_read_sql(*args, num_retries=1, **kwargs):

    try:
        return pd.read_sql(*args, **kwargs)
    except (django.db.utils.InterfaceError,
            django.db.utils.OperationalError) as e:
        connection.close()
        if num_retries == 0:
            raise e
        else:
            return retry_read_sql(*args, num_retries=num_retries-1, **kwargs)


present_device_ids = retry_read_sql("""
SELECT DISTINCT(id) FROM thingapi_rawreading WHERE timestamp > '{}'
""" .format(start_time), connection).values.ravel()


laqn_site_codes = ['HK6', 'CT3']
laqn_sites_df = retry_read_sql("""
SELECT
    site_name,
    site_code
FROM thingapi_laqnsite
WHERE site_code IN (%s)""" % ",".join([
    "'%s'" % code
    for code in laqn_site_codes
]), connection)

calibrated_device_ids = calib.index.values

device_ids = set(present_device_ids).intersection(set(calibrated_device_ids))

print(present_device_ids)
print(calibrated_device_ids)
print(device_ids)


class LaqnStream(streamz.dataframe.DataFrame):

    def __init__(self, start_time, site_code):
        self.current_time = start_time
        self.site_code = site_code
        self.stream = streamz.Stream()
        super().__init__(self.stream, example=self.gen_example())

    def query(self):
        return """
SELECT
    timestamp,
    value AS no2
FROM thingapi_laqnreading
WHERE timestamp > '{}'
AND site_code = '{}'
AND species_code = 'NO2'
ORDER BY timestamp ASC
-- LIMIT 100
""".format(self.current_time, self.site_code)

    def get_data(self):
        df = retry_read_sql(self.query(), connection)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')

        if df.shape[0] > 0:
            self.current_time = df.index[-1]
            print("Received {} LAQN events, taking us up to {}".format(
                df.shape[0],
                self.current_time))
        return df

    def gen_example(self):
        df = pd.DataFrame({'timestamp': [], 'no2': []})
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
        return df

    def callback(self):
        df = self.get_data()
        if df.shape[0] > 0:
            self.stream.emit(df)


class RawReadingStream(streamz.dataframe.DataFrame):

    def __init__(self, start_time, device_id):
        self.current_time = start_time
        self.device_id = device_id
        self.stream = streamz.Stream()
        super().__init__(self.stream, example=self.gen_example())

    def query(self):
        return """
SELECT
    timestamp,
    1 AS intercept,
    no2_w,
    no2_a,
    m_no2,
    CASE WHEN humidity < 95 AND humidity > 30 THEN humidity ELSE (
        SELECT humidity
        FROM thingapi_rawreading
        WHERE id = {}
          AND humidity < 95 AND humidity > 30
        ORDER BY timestamp DESC
        LIMIT 1
    ) END,
    CASE WHEN temp < 80 AND temp > -30 THEN temp ELSE (
        SELECT temp
        FROM thingapi_rawreading
        WHERE id = {}
          AND temp < 80 AND temp > -30
        ORDER BY timestamp DESC
        LIMIT 1
    ) END
FROM thingapi_rawreading
WHERE timestamp > '{}'
--  AND temp > -30
--  AND temp < 80
--  AND humidity < 95
--  AND humidity > 30
AND id = {}
ORDER BY timestamp ASC
""".format(self.device_id, self.device_id, self.current_time, self.device_id)

    def get_data(self):
        df = retry_read_sql(self.query(), connection)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')

        df["no2"] = np.dot(df.values, calib.loc[self.device_id].values)
        df = df[["no2"]]

        if df.shape[0] > 0:
            self.current_time = df.index[-1]
            print("Received {} events, taking us up to {}".format(df.shape[0], self.current_time))
        return df

    def gen_example(self):
        df = pd.DataFrame({'timestamp': [], 'no2': []})
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
        return df

    def callback(self):
        df = self.get_data()
        if df.shape[0] > 0:
            self.stream.emit(df)


dmaps = []
streams = []


for device_id in device_ids:

    original_sdf = RawReadingStream(start_time, device_id)
    sdf = original_sdf.rolling('1h').mean()

    buff = Buffer(sdf, length=60*24)

    dmap = hv.DynamicMap(hv.Curve, streams=[buff])
    dmap = dmap.redim.range(no2=(0, 100))
    dmap = dmap.redim.label(timestamp='Time')
    dmap = dmap.redim.label(no2='no2 (μg/m³)')
    dmap = dmap.relabel("NO2 (Device %d)" % device_id)
    dmap = dmap.opts(plot={"sizing_mode": "stretch_both", "tools": ["hover"]})

    dmaps.append(dmap)
    streams.append(original_sdf)


for index, site in laqn_sites_df.iterrows():

    print(site)

    code = site['site_code']
    name = site['site_name']

    sdf = LaqnStream(start_time, code)

    buff = Buffer(sdf, length=24)

    dmap = hv.DynamicMap(hv.Curve, streams=[buff])
    dmap = dmap.redim.range(no2=(0, 100))
    dmap = dmap.redim.label(timestamp='Time')
    dmap = dmap.redim.label(no2='no2 (μg/m³)')
    dmap = dmap.relabel("NO2 (LAQN - %s)" % name)
    dmap = dmap.opts(plot={"sizing_mode": "stretch_both", "tools": ["hover"]}, style={"color": "grey", "line_width": 5})

    dmaps.append(dmap)
    streams.append(sdf)

eu_max_average_yearly_no2 = hv.HLine(40).opts(plot={"tools": ["hover"]}, style={"color": "grey", "line_width": 1}).relabel("EU max yearly average")
eu_max_average_daily_no2 = hv.HLine(50).opts(plot={"tools": ["hover"]}, style={"color": "grey", "line_width": 1}).relabel("EU max daily average (35*per year)")
eu_max_average_hourly_no2 = hv.HLine(200).opts(plot={"tools": ["hover"]}, style={"color": "grey", "line_width": 1}).relabel("EU max hourly average (18*per year)")
plot = reduce(mul, dmaps)*eu_max_average_yearly_no2*eu_max_average_daily_no2*eu_max_average_hourly_no2


def callback():
    for sdf in streams:
        sdf.callback()


doc = renderer = hv.renderer('bokeh').server_doc(plot)


doc.add_periodic_callback(callback, 1000)
