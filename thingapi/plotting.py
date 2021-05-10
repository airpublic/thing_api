import bokeh
import bokeh.plotting
from bokeh.plotting import gmap
from bokeh.models.sources import ColumnDataSource, AjaxDataSource
from bokeh.embed import components
import bokeh.io
from bokeh.models.widgets.tables import DataTable, TableColumn, NumberFormatter, DateFormatter, StringFormatter

from bokeh.models import (
          GMapPlot, GMapOptions, Circle, Range1d, PanTool, WheelZoomTool, BoxSelectTool
          )

import pandas
from django.db import connection
import time

TS_QUERY="""
WITH
all_readings_date_count AS
(SELECT timestamp::date AS date, COUNT(*) AS c FROM (SELECT timestamp FROM thingapi_rawreading) u GROUP BY timestamp::date ),
date_bounds AS
(SELECT MIN(date) AS min, MAX(date) AS max FROM all_readings_date_count)
, time_series AS
(SELECT generate_series(db.min, db.max, '1 day'::interval) AS date FROM date_bounds db)
SELECT ts.date AS date, COALESCE(c,0) AS n FROM time_series ts LEFT OUTER JOIN all_readings_date_count arc ON ts.date = arc.date
"""

TODAY_QUERY_ALL="""
SELECT
m_no2, timestamp, id
FROM thingapi_rawreading WHERE timestamp > NOW() - interval '1 day'
"""

RECENT_QUERY_ALL="""
SELECT
r.altitude,
r.m_co AS m_co,
r.m_no2 AS m_no2,
r.co_a AS co_a,
r.co_w AS co_w,
r.gps_fix,
r.heading,
r.humidity,
r.latitude,
r.longitude,
r.location_std AS location_std,
r.no2_a AS no2_a,
r.no2_w AS no2_w,
r.pm1,
r.pm10,
r.pm2_5,
r.w_pm1,
r.w_pm10,
r.w_pm2_5,
r.sample_flow_rate,
r.sampling_period,
r.speed,
r.temp,
r.date,
r.lonet_bat,
r.bat_v AS bat_v,
r.retries,
r.timestamp,
r.id,
r.last_organicity_sync,
r.pt
FROM thingapi_rawreading r
{}
ORDER BY r.timestamp DESC
"""

RECENT_QUERY = RECENT_QUERY_ALL + "LIMIT {}"

DEVICE_STATUS_QUERY = """
WITH global_stats AS (
SELECT
id AS id,
MAX(timestamp) AS last_reading,
COUNT(*) AS total
FROM thingapi_rawreading
GROUP BY id
),
last_hour_stats AS (
        SELECT gs.id AS id, COUNT(*) AS total_last_hour, SUM(retries) AS retries_last_hour FROM global_stats gs LEFT OUTER JOIN thingapi_rawreading r ON gs.id = r.id AND r.timestamp > gs.last_reading - '1 hour'::interval GROUP BY gs.id
        ),
last_day_stats AS (
        SELECT gs.id AS id, COUNT(*) AS total_last_day, SUM(retries) AS retries_last_day FROM global_stats gs LEFT OUTER JOIN thingapi_rawreading r ON gs.id = r.id AND r.timestamp > gs.last_reading - '1 day'::interval GROUP BY gs.id
        ),
latest_stats AS (
        SELECT gs.id AS id, lonet_bat AS lonet_bat, bat_v AS bat_v, gps_fix AS gps_fix, r.longitude AS longitude, r.latitude AS latitude FROM global_stats gs INNER JOIN thingapi_rawreading r ON gs.id = r.id AND r.timestamp = gs.last_reading GROUP BY gs.id, lonet_bat, bat_v, gps_fix, r.longitude, r.latitude
        )
SELECT gs.id, gs.last_reading, ls.lonet_bat, ls.bat_v AS bat_v, ls.gps_fix, lh.retries_last_hour, ld.retries_last_day, lh.total_last_hour, ld.total_last_day, ls.longitude AS longitude, ls.latitude AS latitude FROM global_stats gs INNER JOIN last_hour_stats lh ON gs.id = lh.id INNER JOIN last_day_stats ld ON gs.id = ld.id INNER JOIN latest_stats ls ON ls.id = gs.id ORDER BY last_reading DESC;
"""

df_names1 = ["timestamp", "id", "retries", "lonet_bat", "bat_v", "gps_fix", "heading", "speed", "longitude", "latitude", "altitude", "location_std"]
df_names2 = ["timestamp", "id", "humidity", "temp", "m_no2", "m_co", "no2_w", "no2_a", "co_w", "co_a", "pt", "pm1", "pm2_5", "pm10", "sample_flow_rate", "sampling_period", "w_pm1", "w_pm2_5", "w_pm10"]

df_names = list(set(df_names1).union(set(df_names2)))

def readings_per_day():

    p = bokeh.plotting.figure(x_axis_type="datetime", responsive=True, height=300, width=600)

    t0 = time.time()
    df = pandas.read_sql_query(TS_QUERY, connection)
    t1 = time.time()

    print("TS_QUERY in {}s".format(t1-t0))

    ts = p.line(source=ColumnDataSource(df), x="date", y="n")

    return components(p)

def latest_readings_data_source():

    t0 = time.time()
    df = pandas.read_sql_query(RECENT_QUERY.format("", 10), connection)
    t1 = time.time()
    df["timestamp"] = df.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S %f")

    missing = list(set(df.columns.values) - set(df_names))

    print("RECENT_QUERY in {}s".format(t1-t0))

    df["color"] = df["id"].map(pandas.Series({2: "red", 3: "yellow", 4: "green", 5: "white"}))

    return ColumnDataSource(df[df_names])

def device_status():
    t0 = time.time()
    df = pandas.read_sql_query(DEVICE_STATUS_QUERY)
    t1 = time.time()

    print("DEVICE_STATUS_QUERY in {}s".format(t1-t0))

def latest_readings():

#    p = bokeh.plotting.figure(x_axis_type="datetime", sizing_mode="scale_width")


    widths = {"id": 15, "gps_fix": 15, "lonet_bat": 20, "bat_v": 20, "retries": 15, "timestamp": 180, "longitude": 80, "latitude": 80}

    formatters = {"timestamp": StringFormatter()}


    #source = AjaxDataSource(data_url='/v1/latest', polling_interval=1000, mode="replace", method="GET")
    source = latest_readings_data_source()

    columns1 = [TableColumn(field=f, title=f, formatter=formatters.get(f, StringFormatter()), width=widths.get(f, 60)) for f in df_names1]
    columns2 = [TableColumn(field=f, title=f, formatter=formatters.get(f, StringFormatter()), width=widths.get(f, 60)) for f in df_names2]

    ts1 = DataTable(source=source, columns=columns1, width=1200)
    ts2 = DataTable(source=source, columns=columns2, width=1200)

    script1, div1 = components(ts1)
    script2, div2 = components(ts2)

    return (script1+script2, div1+div2)

def map_plot():

#    source = AjaxDataSource(data_url='/v1/latest', polling_interval=1000, mode="replace", method="GET")
    source = device_statuses_data_source()

    map_options = GMapOptions(lat=51.505882, lng=-0.112195, map_type="roadmap", zoom=15)

    plot = gmap(map_options=map_options, title="Recent readings", google_api_key="AIzaSyCHmRvPSlQlj35aICvf8gY__V1y-al4yvk")

    circle = Circle(x="longitude", y="latitude", size=15, fill_color="red", line_color=None)

    plot.add_glyph(source, circle)

#    plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())

    return components(plot)

#    return ("", "")

def device_statuses_data_source():

    t0 = time.time()
    df = pandas.read_sql_query(DEVICE_STATUS_QUERY.format(""), connection)
    t1 = time.time()

    print("DEVICE_STATUS_QUERY in {}s".format(t1-t0))

    df["last_reading"] = df.last_reading.dt.strftime("%Y-%m-%d %H:%M:%S %f")
    df["last_reading"] = df.last_reading

    return ColumnDataSource(df)

def device_statuses():

#    source = AjaxDataSource(data_url='/v1/devices', polling_interval=1000)
#    source = AjaxDataSource(data_url='/v1/devices', polling_interval=1000, mode="replace", method="GET")
    source = device_statuses_data_source()

    df_names = ['id', 'last_reading', 'lonet_bat', 'bat_v', 'gps_fix', 'retries_last_hour', 'retries_last_day', 'total_last_hour', 'total_last_day']

    columns = [TableColumn(field=f, title=f) for f in df_names]

    ds = DataTable(source=source, columns=columns, width=1200)

    return components(ds)

def readings_today_data_source():

    df = pandas.read_sql_query(TODAY_QUERY_ALL, connection)
    df["timestamp"] = df.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S %f")

#    from IPython import embed; embed()
    ts = df.set_index(["id", "timestamp"]).unstack().fillna(0)

#    df["color"] = df["id"].map(pandas.Series({2: "red", 3: "yellow", 4: "green", 5: "white"}))

    return ColumnDataSource(ts)

def readings_today():

    p = bokeh.plotting.figure(x_axis_type="datetime", responsive=True, height=300, width=600)

    t0 = time.time()
    df = pandas.read_sql_query(TODAY_QUERY_ALL, connection)
    t1 = time.time()

    print("TS_QUERY in {}s".format(t1-t0))
    print(df)
    ts = p.line(source=ColumnDataSource(df), x="date", y="n")

    return components(p)
