import holoviews as hv
import geoviews as gv
import geoviews.tile_sources as gvts
from holoviews.streams import Buffer, Stream
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import reduce
from operator import mul
from bokeh.io import curdoc
from operator import __mul__, __add__
from cartopy import crs
from bokeh.models.tools import TapTool, HoverTool
from holoviews.streams import Tap, Selection1D
from bokeh.themes import Theme
from bokeh.transform import factor_cmap
from pytz.exceptions import NonExistentTimeError, AmbiguousTimeError
from dateutil.parser import isoparse
from bokeh.models.tickers import FixedTicker
#from fullscreentool import FullscreenTool
from holoviews.operation.datashader import datashade
import param
import panel as pn
import json
import functools
import methodtools
from bokeh.models.tools import PanTool, WheelZoomTool, ResetTool
from holoviews import opts

hv.extension('bokeh')
gv.extension('bokeh')

import sys, os
sys.path.append('.')
os.environ['DJANGO_SETTINGS_MODULE'] = os.environ.get('DJANGO_SETTINGS_MODULE', 'thingapi.settings')
import django
django.setup()


from django.db import connection

from thingapi.models import Experiment, DeviceTimeRange, WeatherDataSource
from thingapi.kcl_colors import kcl_no2_colors, kcl_no2_levels, kcl_clipping_colors
from thingapi.loaders import AbstractLoader, NO2_SPECIES_COLUMN

from bokeh.models import Select, Div
from bokeh.layouts import row, widgetbox, column, layout
from datetime import datetime
import dateutil

from tornado import gen
from bokeh.document import without_document_lock
from concurrent.futures import ThreadPoolExecutor
from collections import namedtuple


hv.config.future_deprecations = True


wind_rose_option_names = {
    "wind": "Wind",
    "pollution": "Pollution"
}


time_series_hover = HoverTool(
    tooltips=[
        ('Date & Time', '@timestamp{%F %T}'),
        ('Name/number', '@device_id'),
        ('Species', '@species'),
        ('Value', '@value{0.0 a} @unit'),
    ],
    formatters={
        '@timestamp': 'datetime'
    }
)


map_hover = HoverTool(
    tooltips=[
# See https://github.com/holoviz/geoviews/issues/251
#        ('Latitude', "@latitude"),
#        ('Longitude', "@longitude"),
        ('Location', "@latlon"),
        ('Date & Time', "@timestamp{%F %T}"),
        ('Name/Number', "@device_id"),
        ('Species', '@species'),
        ('Value', '@value{0.0 a} @unit')
    ],
    formatters={
        '@timestamp': 'datetime'
    }
)


def disable_bokeh_logo(plot, element):
    plot.state.toolbar.logo = None


def fix_tools_for_alert(plot, element):
    tools_by_type = {}

    # Dedupe tools by type, but only the ones we want
    for tool in plot.state.toolbar.tools:
        if isinstance(tool, (PanTool, WheelZoomTool, ResetTool)):
            tools_by_type[str(type(tool))] = tool

    plot.state.toolbar.tools = list(tools_by_type.values())


class ProxyLoader(AbstractLoader):

    def __init__(self, dashboard):
        self.dashboard = dashboard

    def results_as_dataframe(self):
        return self.dashboar.calibrated_results

    def get_species(self):
        return self.dashboard.species

    def get_device_ids(self):
        return self.dashboard.device_ids

    def postprocess(self, df):
        return df


class Dashboard():

    def __init__(self, experiment_id):
        self.client_timezone_name = 'Europe/London'
        self.client_timezone = dateutil.tz.gettz(self.client_timezone_name)
        self.client_now = datetime.now(self.client_timezone)
        self.client_timezone_static = dateutil.tz.tzoffset(
                name=self.client_timezone.tzname(self.client_now),
                offset=self.client_timezone.utcoffset(self.client_now))
        self.time_resolution = '1h'
        self.experiment_id = experiment_id
        self.calibrated_results = DeviceTimeRange.empty_result_df()
        self.calibrated_results["timestamp"] = (
                self.calibrated_results["timestamp"]
                .dt.tz_convert(self.client_timezone)
                .dt.tz_localize(None)
        )

    def get_alert_condition(self, id):
        return self.__alert_conditions__[int(id)]

#    @property
#    def alerts(self):
#        self.ensure_data_loaded()
#        return self.__alerts__
        return []

    @property
    def experiment(self):
        return self.__experiment_method__()


    @methodtools.lru_cache()
    def __experiment_method__(self):
        return Experiment.objects.filter(id=self.experiment_id).first()

    def ensure_data_loaded(self):
        try:
            self.data_loaded
        except AttributeError:
            self.load_data()

    def get_loader_placeholder_data(self, loader=None, species=None, device_ids=None):
        """
        Create a dataframe consisting of a single dummy data point per device
        per species. Either loader must be passed or species and device_ids
        must be passed. This dummy data is used to get round various bugs in
        bokeh and holoviews
        """
        if species is None:
            species = loader.get_species()

        if device_ids is None:
            device_ids = loader.get_device_ids()


        def safe_getter(df, key):
            def inner(d_id):
                try:
                    return df[key].loc[d_id]
                except KeyError:
                    # Most likely caused by no data for this device
                    return np.nan
            return inner

        def always_nan(d_id):
            return np.nan

        if loader:
            location_df = loader.get_dummy_locations()
            lat = safe_getter(location_df, "latitude")
            lon = safe_getter(location_df, "longitude")
        else:
            lat, lon = always_nan, always_nan

        df = pd.DataFrame([
            [
                lat(device_id),
                lon(device_id),
                self.initial_timestamp.tz_convert("UTC"),
                device_id,
                sp,
                "Not loaded",
                0.0
            ]
            for sp in species
            for device_id in device_ids
        ], columns=["latitude", "longitude", "timestamp", "device_id", "species", "unit", "value"])

        return df

    def reify_timezones_inplace(self, df):
        """
        Holoviews converts all timestamps to UTC before plotting. To show data in
        the correct timezone and get daylight savings time right, we convert to the
        client timezone, then discard all time zone info. Holoviews will then assume
        our timezones are UTC and do no conversion. We will see jumps, or wraps at
        the DST transitions.
        https://gist.github.com/johncant/9bc0048079b49351c2e22be872591fed

        This method does this to the timestamp column of the given dataframe inplace
        """
        df["timestamp"] = (
            df["timestamp"]
            .dt.tz_convert(self.client_timezone_name)
            .dt.tz_localize(None)
        )

    def load_data(self):

        calibrated_results = self.experiment.load_data()

        self.reify_timezones_inplace(calibrated_results)

        self.calibrated_results = calibrated_results
        self.load_alerts()
        self.data_loaded = True

    def load_alerts(self):
        pass
#        self.__alerts__ = list(self.experiment.alerts.all())
#        self.__alert_conditions__ = {}
#
#        executor = ThreadPoolExecutor(max_workers=30)
#
#        for alert in self.__alerts__:
#            condition = alert.instantiate_condition()
#
#            condition_data = condition.should_alert(self.proxy_loader, self.calibrated_results)
#
#            # This fixes a bug in holoviews/numpy where hv attempts to subtract
#            # booleans, which is banned by numpy
#            condition_data["should_alert"] = condition_data["should_alert"].astype(int)
#
#            self.__alert_conditions__[alert.id] = condition_data

    @property
    @methodtools.lru_cache()
    def proxy_loader(self):
        return ProxyLoader(self)

    def load_metadata(self):

        executor = ThreadPoolExecutor(max_workers=30)

        weather_data_sources = list()
        weather_data_source_loaders = {
        }

        for wds in WeatherDataSource.objects.all():
            weather_loader = wds.instantiate_loader()
            for name in weather_loader.site_list():
                weather_data_sources.append(name)
                weather_data_source_loaders[name] = wds.instantiate_loader()

        self.weather_data_sources = weather_data_sources
        self.weather_data_source_loaders = weather_data_source_loaders

        self.species = sorted(list(set(
            species
            for loader in self.experiment.get_loaders()
            for species in loader.get_species())))



        self.initial_species = list(self.species)[0]

        self.device_ids = [
            id
            for loader in self.experiment.get_loaders()
            for id in loader.get_device_ids()
        ]

        print(self.initial_species)

        self.initial_weather_data_source = weather_data_sources[0]

        # Insert dummy data to overcome bug where setting axis range is broken
        self.initial_timestamp = pd.Timestamp.now().tz_localize(self.client_timezone_name)
        self.calibrated_results = pd.concat([
            self.get_loader_placeholder_data(loader)
            for loader in self.experiment.get_loaders()
        ])
        self.reify_timezones_inplace(self.calibrated_results)

    def get_species_data(self, species_):
        species_results = self.calibrated_results[self.calibrated_results['species']==species_].sort_values('timestamp')
        return species_results

    def sanitize_timestamp(self, ts_str_or_ts):

        # This is the time zone the caller wants in the result.
        # If none, we will assume client timezone
        if isinstance(ts_str_or_ts, str):
            input_tz = isoparse(ts_str_or_ts).tzinfo
            ts = pd.to_datetime(ts_str_or_ts)
        elif isinstance(ts_str_or_ts, pd.Timestamp):
            ts = ts_str_or_ts
            input_tz = ts.tz
        elif isinstance(ts_str_or_ts, np.datetime64):
            ts = pd.to_datetime(ts_str_or_ts)
            input_tz = None # These cannot be timezone aware
        else:
            raise ValueError(("""\
    sanitize_timestamp only takes the following types:
      - str
      - pandas.TimeStamp
      - numpy.datetime64

    Received {}
    """).format(type(ts_str_or_ts)))

        if input_tz is None:
            # Naive timezone assumed to be in client timezone
            input_tz = self.client_timezone_name
            output_tz = self.client_timezone_name
        else:
            # Pandas has converted to UTC and discarded info
            output_tz = input_tz
            input_tz = 'UTC'

        ts_rounded = ts.round(self.time_resolution)

        def attempt_localization(ts):
            if ts.tz:
                return ts.tz_convert(output_tz)
            else:
                return ts.tz_localize(input_tz).tz_convert(output_tz)

        try:
            return attempt_localization(ts_rounded)
        except NonExistentTimeError:
            # The user selected a time during the transition from normal
            # time to daylight savings time. i.e. in Europe/London, this
            # could be 2018-03-25 01:30:00

            # Assume only 1 hour was missed - if not, we might have to do a kind
            # of search to get to the closest valid timestamp

            round_diff = ts - ts_rounded
            extra_diff = np.sign(round_diff.total_seconds())*pd.to_timedelta('1 hour')

            ts_rounded1 = (ts + extra_diff).round(self.time_resolution)

            return attempt_localization(ts_rounded1)
        except AmbiguousTimeError:
            # The user selected a time during the transition from daylight savings
            # time to normal time, i.e. in Europe/London, this could be
            # 2017-10-29 01:30:00 . The goal here is really just to select any
            # of them since with the time zone information discarded, data points
            # at both times should match and should show on the map.

            # Subtract an hour, do the conversion, then add an hour.

            hacked_ts = ts_rounded - pd.to_timedelta('1 hour')
            hacked_ts_local = attempt_localization(hacked_ts)
            return hacked_ts_local + pd.to_timedelta('1 hour')

    def timestamp_to_bokeh(self, ts):
        return ts.tz_convert(self.client_timezone).tz_localize(None)

    def get_joined_weather_pollution(self, species, timestamp,
            weather_data_source_name, device_ids):

        loader = self.weather_data_source_loaders[weather_data_source_name]

        timestamp = dashboard.sanitize_timestamp(timestamp).tz_convert('UTC')

        end_timestamp = timestamp
        start_timestamp = end_timestamp - pd.to_timedelta('7d')

        weather = loader.site_weather_dataframe(weather_data_source_name)
        pollution = self.get_species_data(species)
        pollution = pollution[pollution.device_id.isin(device_ids)]

        weather = weather.to_period('1h')
        pollution = pollution.set_index('timestamp', drop=True).to_period('1h')

        weather = weather[(weather.index >= start_timestamp) & (weather.index < end_timestamp)]
        pollution = pollution[(pollution.index >= start_timestamp) & (pollution.index < end_timestamp)]

        # This should deal with 3h weather
        weather = weather.resample(pollution.index.freq).ffill()

        joined_df = weather.merge(pollution, left_index=True, right_index=True)
        return joined_df


class PollutionWindRose(namedtuple("PollutionWindRose", ["dashboard"])):

    def dir_labels(self):
        return ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    def fix_angle_range(self, series):
        return ((series + 11.25) % 360) - 11.25

    def get_grouped_data(self, df):
        spd_bins = np.arange(0, 20, 4)
        spd_labels = spd_bins[0:-1]

        dir_bins = np.arange(-11.25, 370, 22.5)
        dir_labels = dir_bins[0:-1]

        return df.assign(
            WindDir_bins=lambda df: pd.cut(self.fix_angle_range(df['WindDir']), bins=dir_bins, labels=dir_labels,
                                                 right=False)
        ).assign(
            WindSpd_bins=lambda df: pd.cut(df['WindSpd'], bins=spd_bins, labels=spd_labels, right=False)).replace(
            {'WindDir_bins': {360: 0}}
        ).groupby(by=['WindSpd_bins', 'WindDir_bins'])

    def get_pollution_rose_data(self, df):
        grouped = self.get_grouped_data(df)
        g1 = grouped.count()['WindDir']
        g2 = grouped.mean()['value']
        g = pd.DataFrame({'WindDir': g1, 'value': g2})
        return g.reset_index()

    def get_graph(self, timestamp=None, weather_data_source=None, device_ids=None,
                  species=None, wind_rose_mode=None):

        wind_dir = hv.Dimension('WindDir_bins', label="Wind direction")
        wind_spd = hv.Dimension('WindSpd_bins', label="Wind speed", unit='knots')
        species_dim = hv.Dimension('value', label='7 day average') # TODO - units dashboard-wide
        samples = hv.Dimension('WindDir', label="Samples")

        if wind_rose_mode == wind_rose_option_names["wind"]:
            vdims = [samples, species_dim]
        elif wind_rose_mode == wind_rose_option_names["pollution"]:
            vdims = [species_dim, samples]
        else:
            raise ValueError("Unrecognised wind rose option {}".format(wind_rose_mode))

        joined_df = dashboard.get_joined_weather_pollution(
                timestamp=timestamp, device_ids=device_ids,
                weather_data_source_name=weather_data_source, species=species)
        g = self.get_pollution_rose_data(joined_df)

        hm = hv.HeatMap(
            g,
            kdims=[wind_dir, wind_spd],
            vdims=vdims,
            cdims={'WindDir': samples}
        ).options(
#            tools=['hover', FullscreenTool()],
            tools=['hover'],
            radial=True,
            xticks=self.dir_labels(),
            xmarks=len(self.dir_labels()),
            start_angle=np.pi * 5 / 8,
            colorbar=True,
        )

        print(joined_df.shape)

        if wind_rose_mode == wind_rose_option_names["pollution"] and species in {'NO2_ug_per_m3', 'no2', NO2_SPECIES_COLUMN, 'PM10'} and joined_df.shape[0] > 0:
            hm = hm.options(
                cmap=kcl_no2_colors,
                color_levels=kcl_no2_levels,
#                color='value',
                clipping_colors=kcl_clipping_colors,
                colorbar=True,
                clim=(0, 100)
            )

        else:
            hm = hm.options(
#                color='value',
                cmap='viridis',
                colorbar=True,
            )

        return hm


def extract_argument(key):
    ctx = curdoc().session_context

    if ctx:
        bokeh_request_arguments = ctx.request.arguments
    else:
        bokeh_request_arguments = json.loads(os.environ.get('BOKEH_REQUEST_ARGUMENTS'))

    argument = bokeh_request_arguments.get(key)

    if isinstance(argument, list):
        return argument[0]
    else:
        return argument

experiment_id = extract_argument("experiment_id")
dashboard = Dashboard(experiment_id)
dashboard.load_metadata()
dashboard.load_data()

pollution_wind_rose = PollutionWindRose(dashboard)


def time_dimension():
    return hv.Dimension("timestamp", label="Date & Time (local)", type=pd.datetime)




def create_device_timeseries_figure3(species, device_ids, timestamp=None, weather_data_source=None):

    value_dimension = hv.Dimension("value", label=species)
    device_dimension = hv.Dimension("device_id", label="Name/Number")

    dashboard.ensure_data_loaded()

    species_results = dashboard.get_species_data(species)
    device_results = species_results[species_results["device_id"].isin(device_ids)]

    print(device_results.head())

    hv_table = hv.Table(device_results, [
        time_dimension(),
        'species',
        value_dimension,
        device_dimension,
        'unit'
    ])
    lines = hv_table.to(hv.Curve, time_dimension(), value_dimension, groupby=device_dimension).overlay()
    points = hv_table.to(hv.Points, [time_dimension(), value_dimension], ['species', 'unit'], groupby=device_dimension).overlay()

    points = points.opts(
        opts.Points(tools=["tap"], size=5)
    )

    points = points.opts(
        opts.Points(tools=[time_series_hover], size=5)
    )
#        'Points': {
#            "tools": [
#                "tap",
#                time_series_hover
#            ],
#            'size': 5,
#        }
#    })


    plot = points*lines

    return plot


global extents_previously_unavailable
extents_previously_unavailable = False

def create_map_figure(species, timestamp=None, **kwargs):
    global extents_previously_unavailable
    _device_ids = kwargs.get('device_ids', None)
    dashboard.ensure_data_loaded()

    if timestamp:
        timestamp = dashboard.sanitize_timestamp(timestamp)

        # In the dataset, we removed time zone information to get the correct
        # behaviour from holoviews. But before we did that, we localized it to
        # the client timezone. `timestamp` is now not guaranteed to be
        # in any particular timezone. But it is guaranteed to be time zone
        # aware

        timestamp = dashboard.timestamp_to_bokeh(timestamp)

    if _device_ids:
        _device_ids = set(_device_ids)
    else:
        _device_ids = dashboard.device_ids

    species_results = dashboard.get_species_data(species)

    this_time_results = species_results[species_results["timestamp"]==timestamp]

    unshown_device_ids = set(dashboard.device_ids) - set(this_time_results.device_id)

    # Show most recent positions in gray
    missing_species_results = species_results[species_results["device_id"].isin(unshown_device_ids)]
    missing_species_results = missing_species_results.groupby(by=['device_id']).apply(lambda df: df.sort_values('timestamp', ascending=False).iloc[0:1])
    missing_species_results = missing_species_results.copy()
    missing_species_results["value"] = np.nan

    this_time_results = pd.concat([this_time_results, missing_species_results])

    tiles = gvts.ESRI # .options(max_zoom=5, min_zoom=15)

    sum_position_stds = this_time_results[['longitude', 'latitude']].std().sum()
    extents_unavailable = (not sum_position_stds or sum_position_stds < 1e-10)
    extents_need_changing = False

    extra_kwargs = {
#        "colorbar": True
    }

    # GeoViews can't deal with this case and starts
#    if extents_unavailable:
#        extra_kwargs["extents"] = (-180, -90, 180, 90)
#    else:
#        pass
#        data_extents = (
#            this_time_results.longitude.min(),
#            this_time_results.latitude.min(),
#            this_time_results.longitude.max(),
#            this_time_results.latitude.max()
#        )
#        span_x = data_extents[2] - data_extents[0]
#        span_y = data_extents[3] - data_extents[1]
#        padding_x, padding_y = np.maximum(span_x, 1e-3), np.maximum(span_y, 1e-3)
#
#        extents = (
#            data_extents[0] - padding_x,
#            data_extents[1] - padding_y,
#            data_extents[2] + padding_x,
#            data_extents[3] + padding_y,
#        )
#
#        extra_kwargs["extents"] = extents


#        extra_kwargs["colorbar"] = False

    extra_kwargs["crs"] = tiles.crs # This is necessary to fix aspect ratio

    this_time_results['latlon'] = this_time_results['latitude'].astype('str') + ',' + this_time_results['longitude'].astype('str')

    points = gv.Points(this_time_results,
                       kdims=['longitude', 'latitude'],
                       vdims=[
                           hv.Dimension('value', range=(kcl_no2_levels[0], kcl_no2_levels[-1])),
                           'species',
                           time_dimension(),
                           'device_id',
                           'unit',
                           'latlon'
                       ],
                       **extra_kwargs
                       )

    labels = gv.Labels(this_time_results[['longitude', 'latitude', 'device_id']]).options(text_color='white', yoffset=30)

    if species in {'NO2_ug_per_m3', 'no2', NO2_SPECIES_COLUMN, 'PM10'}:
        points = points.options(
            cmap=kcl_no2_colors,
            color_levels=kcl_no2_levels,
            color='value',
            clipping_colors=kcl_clipping_colors,
            colorbar=True,
            clim=(0, 100),
            size=20,
            tools=[map_hover],
        )

    else:
        points = points.options(
            color='value',
            cmap='viridis',
            colorbar=True,
            size=20,
            tools=[map_hover]
        )

    points = points*labels
    points = points.options(framewise=True)

    plot = (tiles*points)
    plot = plot.options(framewise=True)

    extents_previously_unavailable = extents_unavailable

    return plot


def create_vline(timestamp=None, y=None, **kwargs):

    if pd.isnull(timestamp):
        timestamp = dashboard.initial_timestamp
#    # When we wrote this, the timestamp was time zone naive. However,
#    # upgrading from Pandas 0.23.4 to 0.24.0 caused the timestamp here
#    # to start to be time zone aware. Solution: Discard time zone.
#    timestamp = pd.to_datetime(timestamp).tz_localize(None)


    timestamp = dashboard.sanitize_timestamp(timestamp)

    # To avoid Holoviews converting the time from our time zone to UTC, we
    # remove the time zone information
    timestamp = dashboard.timestamp_to_bokeh(timestamp).asm8

    return hv.VLine(timestamp).options(color='red')


#species_select = Select(title='Species',
#                        value=None,
#                        options=['None']+list(dashboard.species))


class ParamClass(Stream):
    species = param.ObjectSelector(default=dashboard.initial_species, objects=dashboard.species)
    timestamp = param.String(default=dashboard.initial_timestamp.isoformat(), doc="Time slice", constant=False)
    device_ids = param.ListSelector(default=list(dashboard.device_ids), objects=list(dashboard.device_ids),precedence=0.5)
    weather_data_source = param.ObjectSelector(default=dashboard.initial_weather_data_source, objects=dashboard.weather_data_sources)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_streams = []

    def watch_fn(self, change):
        self.event(**{change.name: change.new})

    def event(self, **kwargs):
        super().event(**kwargs)
        for stream in self.sub_streams:
            stream.event(**kwargs)

    def exclude(self, *attrs):
        pass
#        self.sub_streams =

    def rename(self, **kwargs):
        stream = super().rename(**kwargs)
        self.sub_streams.append(stream)
        return stream


#class AlertParamClass(Stream):
#    alerts = param.List(default=dashboard.alerts)


class WindRoseParam(Stream):
    wind_rose_mode = param.ObjectSelector(
                        default=wind_rose_option_names["pollution"],
                        objects=[
                            wind_rose_option_names["wind"],
                            wind_rose_option_names["pollution"]])

    def watch_fn(self, change):
        self.event(**{change.name: change.new})


param_instance = ParamClass(name="")
#alerts_param_instance = AlertParamClass(name="")
wind_rose_params = WindRoseParam(name="Wind rose options")

time_series_param_stream = param_instance.rename(timestamp=None, weather_data_source=None)


vline = hv.DynamicMap(create_vline, streams=[param_instance])
time_series = hv.DynamicMap(
        create_device_timeseries_figure3,
        streams=[time_series_param_stream]
)


def create_alert_plot(alerts):

    plots = []

    for alert in alerts:

        df = dashboard.get_alert_condition(alert.id)

        df = df[['should_alert', 'timestamp']]

        # Do interpolation myself. hv.Area doesn't support the interpolation
        # argument from hv.Curve

        df['should_alert'] = df['should_alert'].astype(bool)
        df = df.sort_values('timestamp')
        df['is_change'] = df['should_alert'] != df['should_alert'].shift(1)
        df['is_change'].iloc[0] = True

        df = df[df['is_change']]
        df1 = df.assign(actual=1)
        df2 = df.assign(actual=0, should_alert=~df['should_alert'])
        df = pd.concat([df1, df2]).sort_values(['timestamp', 'actual'])
        df['should_alert'] = df['should_alert'].astype(int)

        plots.append(
            hv.Area(df, kdims=['timestamp'], vdims='should_alert', interpolation='steps-post').options(
                height=200,
                responsive=True,
                hooks=[disable_bokeh_logo, fix_tools_for_alert],
                color='red',
                line_width=0,
                bgcolor='green',
                tools=['xpan', 'xwheel_zoom', 'reset']
            )
        )

    if not plots:

        plots.append(Div(text='<div class="alerts_empty"><h1>No alerts defined</h1></div>'))

    return pn.Column(*plots, sizing_mode="stretch_both")


def create_interactive_alert_plots(stream):

    plot = create_alert_plot(stream.alerts)
    parent = pn.Row(plot, sizing_mode="stretch_both")

    def update():
        plot = create_alert_plot()
        parent[0] = plot

    stream.param.watch(update, 'alerts')

    return parent


#alerts_list_view = create_interactive_alert_plots(alerts_param_instance)
#alert_layout = alerts_list_view

#time_series = datashade(time_series)
tap_stream = Tap(source=vline)

def update_param(x=None, **kwargs):
    """
    update all the plots on the click on the timeline graph
    :param x:
    :param kwargs:
    :return:
    """
    if x:
        timestamp = dashboard.sanitize_timestamp(x)

        param_instance.timestamp = timestamp.__str__()

tap_stream.add_subscriber(update_param)

time_series = time_series*vline
time_series = time_series
map_fig = hv.DynamicMap(create_map_figure, streams=[param_instance.rename(weather_data_source=None, device_ids=None)])
wind_rose = hv.DynamicMap(pollution_wind_rose.get_graph, streams=[param_instance, wind_rose_params])


# hook up the watchers
param_instance.param.watch(param_instance.watch_fn, 'species')
param_instance.param.watch(param_instance.watch_fn, 'timestamp')
param_instance.param.watch(param_instance.watch_fn, 'device_ids')
param_instance.param.watch(param_instance.watch_fn, 'weather_data_source')
wind_rose_params.param.watch(wind_rose_params.watch_fn, 'wind_rose_mode')


titles = Div(text=f"""\
<h1><img src="/static/thingapi/logo_white_transparent.png"></h1>
<p><a href="/experiments/">Experiments</a> - {dashboard.experiment.name}</p>
""")
space = Div(text="")

wind_rose_widgets = pn.Column(wind_rose_params)
param_widgets = pn.Param(param_instance.param, widgets={
    'device_ids': pn.widgets.CheckBoxGroup
})
widgets = pn.Column(titles, param_widgets, wind_rose_widgets)

param_widgets = pn.Param(param_instance.param, widgets={
    'device_ids': pn.widgets.CheckBoxGroup
})
widgets = pn.Column(titles, param_widgets, wind_rose_widgets)

time_series = time_series.options(hooks=[disable_bokeh_logo], responsive=True)
map_fig = map_fig.options(hooks=[disable_bokeh_logo], responsive=True)
wind_rose = wind_rose.options(hooks=[disable_bokeh_logo], responsive=True)


def final_output_step(time_series, map_fig):

    main_layout = pn.Row(widgets, pn.Column(
        pn.Row(wind_rose, map_fig, sizing_mode='stretch_both'),
        pn.Row(time_series, sizing_mode='stretch_both'),
        sizing_mode='stretch_both',
    ), sizing_mode='stretch_both')

    layout = main_layout
#    layout = pn.Tabs(
#        ('Overview', main_layout),
#        ('Alerts (Alpha)', alert_layout)
#    )

    doc = layout.servable()

    doc.theme = Theme(
        filename=os.path.join(
            os.path.dirname(__file__),
            "themes.yaml"
        )
    )

    return doc

renderer = hv.renderer('bokeh').instance(mode='server')

if os.environ.get("AIRPUBLIC_UNDER_DEVELOPMENT_FROM_IPYNB", None):
    # If this environment variable is set, we're developing this from a
    # notebook. Don't do any plotting - just define a bunch of stuff for the
    # developer to override from a notebook
    pass
    dashboard.load_data()
else:
    # We are most likely being run from bokeh serve
    doc = final_output_step(time_series, map_fig)


print("Rendering done")
