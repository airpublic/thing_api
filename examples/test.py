import holoviews as hv
from holoviews.streams import Buffer
import pandas as pd
import numpy as np

renderer = hv.renderer('bokeh').instance(mode='server')

df = pd.DataFrame({'x': [], 'y': []}).set_index('y')

stream = Buffer(df, length=30)

dmap = hv.DynamicMap(hv.Curve, streams=[stream]).redim.range(y=(0,1)).opts(plot={"sizing_mode": "stretch_both"})



i = 0
def cb():
    global i

    df = pd.DataFrame({'x': [i], 'y': np.random.rand(1)}).set_index('y')
    i = i + 1
    stream.send(df)


doc = renderer = hv.renderer('bokeh').server_doc(dmap)
doc.add_periodic_callback(cb, 0.05)
