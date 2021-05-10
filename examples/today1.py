import holoviews as hv
from holoviews.streams import Buffer
import pandas as pd
import numpy as np


import os
os.environ['DJANGO_SETTINGS_MODULE'] = os.environ.get('DJANGO_SETTINGS_MODULE', '..thingapi.settings')
import django
django.setup()


from django.database import connection


renderer = hv.renderer('bokeh').instance(mode='server')


stream = Buffer(df, length=30)

dmap = hv.DynamicMap(hv.Curve, streams=[stream]).redim.range(y=(0,1)).opts(plot={"sizing_mode": "stretch_both"})


class RawReadingStream():

    def __init__(self):
        self.current_time = None

    def initial_data(self):
        return pd.read_sql("SELECT timestamp, m_no2 AS no2 FROM thingapi_rawreading", connection)

    def callback(self):
        return pd.read_sql("SELECT timestamp, m_no2 AS no2 FROM thingapi_rawreading WHERE 1 = 0", connection)


rrs = RawReadingStream()
df = rrs.initial_data()

i = 0
def cb():
    global i

    df = pd.DataFrame({'x': [i], 'y': np.random.rand(1)}).set_index('y')
    i = i + 1
    stream.send(df)


doc = renderer = hv.renderer('bokeh').server_doc(dmap)
doc.add_periodic_callback(rrs.callback, 5)
