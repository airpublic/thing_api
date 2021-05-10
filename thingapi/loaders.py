import os
import pickle
import io
import json
from datetime import datetime
from abc import ABC, abstractmethod
import pyarrow.feather
import traceback

import numpy as np
import pandas as pd
from django.db import connection
from psycopg2.extensions import adapt
from concurrent.futures import ThreadPoolExecutor
from django.db.models import Q

from thingapi.models import (
    RawReading, DeviceTimeRange, METSite, RawReadingV4,
    PrecomputedLoaderData)
from thingapi.models import instantiate_by_name_with_args

miles_to_knots = 1.15

NO2_SPECIES_COLUMN = "NO₂"
PM10_SPECIES_COLUMN_IN_DASHBOARD = "PM10"


class AbstractLoader(ABC):
    """
    This class is responsible for loading and running calibration models.
    """

    @abstractmethod
    def __init__(self, device_time_range, *args, **kwargs):
        pass

    @abstractmethod
    def results_as_dataframe(self):
        pass

    def get_device_ids(self):
        return [str(self.dtr.device_id)]

    def get_species(self):
        return [self.dtr.species]

    def postprocess(self, df):
        pass

    def get_dummy_locations(self):
        """
        DataFrame of latitude and longitude indexed by device id
        """
        return pd.DataFrame({
            "device_id": self.get_device_ids(),
            "latitude": np.nan,
            "longitude": np.nan
        }).set_index('device_id')


class CustomDeviceLabelMixin():

    def __init__(self, *args, **kwargs):
        self.device_name = kwargs.get('device_name', None)
        if self.device_name is not None:
            assert type(self.device_name) is str
        super().__init__(*args, **kwargs)

    def postprocess(self, df):
        if self.device_name:
            df["device_id"] = self.device_name
        super().postprocess(df)

    def get_device_ids(self):
        if self.device_name is not None:
            return [self.device_name]
        else:
            return super().get_device_ids()


class CustomLocationMixin():

    def __init__(self, *args, **kwargs):
        self.location = kwargs.get('location', None)
        if self.location is not None:
            assert type(self.location) is list
            assert len(self.location) == 2
        super().__init__(*args, **kwargs)

    def postprocess(self, df):
        if self.location:
            df["latitude"] = self.location[0]
            df["longitude"] = self.location[1]
        super().postprocess(df)

    def get_dummy_locations(self):
        if self.location:
            return pd.DataFrame({
                "device_id": self.get_device_ids(),
                "latitude": self.location[0],
                "longitude": self.location[1]
            }).set_index('device_id')
        else:
            return super().get_dummy_locations()


class Demo(AbstractLoader):
    """
    This loader returns some raw data from v3. Since none of the devices
    transmit at exactly the same time, you can't expect the map to work
    since it looks for an exact timestamp.

    This hasn't been tested in a while
    """

    def __init__(self, device_time_range, *args, **kwargs):
        self.dtr = device_time_range

    def raw_v3_data(self):

        readings = RawReading.objects.all()
        readings = readings.filter(id__exact=self.dtr.device_id)

        if self.dtr.start_timestamp:
            readings = readings.filter(timestamp__gt=self.dtr.start_timestamp)

        if self.dtr.end_timestamp:
            readings = readings.filter(timestamp__gt=self.dtr.end_timestamp)

        query = readings.all().query.__str__()
        data = pd.read_sql(query, connection)
        print(query)

        if data.shape[0] > 0:

            dfs = []

            for value_column in ['temp', 'humidity', 'bat_v', 'm_no2', 'no2_w',
                                 'no2_a']:
                dfs.append(pd.DataFrame({
                    "timestamp": data["timestamp"],
                    "latitude": data["latitude"],
                    "longitude": data["longitude"],
                    "species": value_column,
                    "value": data[value_column],
                    "device_id": self.dtr.device_id,
                    "timestamp": data["timestamp"],
                }))

            result = pd.concat(dfs)
            result["device_id"] = result["device_id"].astype(str)
            return result
        else:
            return DeviceTimeRange.empty_result_df()

    def results_as_dataframe(self):
        df = self.raw_v3_data()
        self.postprocess(df)
        return df


class RawDataSQLLoader(AbstractLoader):
    """
    Load raw data from SQL tables
    """

    @abstractmethod
    def table_name(self):
        pass

    def __init__(self, device_time_range, *args, **kwargs):
        self.dtr = device_time_range
        self.mobile = kwargs.get("mobile", True)
        self.collinear = kwargs.get("collinear", False)
        self.fill_gaps = kwargs.get("fill_gaps", False)
        super().__init__(device_time_range, *args, **kwargs)

    def sql_conditions(self):
        conditions = []

        conditions.append("id = %d" % self.dtr.device_id)

        if self.dtr.start_timestamp:
            conditions.append(
                "timestamp > '%s'" % self.dtr.start_timestamp.isoformat())

        if self.dtr.end_timestamp:
            conditions.append(
                "timestamp < '%s'" % self.dtr.end_timestamp.isoformat())

        return conditions

    def sql_conditions_as_where_clause(self):
        conditions = self.sql_conditions()

        if conditions:
            conditions = "\nWHERE %s" % " AND ".join(conditions)
        else:
            conditions = "\n"

        return conditions

    def get_data_from_sql(self):
        lat_col, lon_col, gps_fix_col = self.gps_column_names()

        lat_sql = "AVG(CASE WHEN %s = 1 THEN %s ELSE NULL END) AS latitude" % (
            gps_fix_col, lat_col
        )

        lon_sql = "AVG(CASE WHEN %s = 1 THEN %s ELSE NULL END) AS longitude" % (
            gps_fix_col, lon_col
        )

        conditions = self.sql_conditions_as_where_clause()

        value_columns_sql = ",\n    ".join(["AVG(%s) AS %s" % (col, col)
                                            for col in self.value_columns])

        query = """
        SELECT
            date_trunc('hour', timestamp) AS timestamp,
            id,
            %s,
            %s,
            %s
        FROM %s %s
        GROUP BY id, date_trunc('hour', timestamp)
        ORDER by date_trunc('hour', timestamp) DESC
        LIMIT 10000
        """ % (lat_sql, lon_sql, value_columns_sql, self.table_name(), conditions)

        print(query)
        data = pd.read_sql(query, connection)

        # This is to prevent pandas dropping columns of all None during resampling
        for col in self.value_columns + ["latitude", "longitude"]:
            data[col] = data[col].astype(np.float32)

        #        if self.fill_gaps:
        print(data.shape)

        return data

    def convert_for_hourly_display(self, data, unit=''):
        if not self.mobile:
            data["latitude"] = np.mean(data["latitude"])
            data["longitude"] = np.mean(data["longitude"])

        if data.shape[0] > 0:
            # Fill gaps with NaN, but only fill them with one NaN either side
            # of valid data ranges. This is to prevent lines joining separate
            # sequences

            dfs = []

            for value_column in self.value_columns:

                data_v = data[[value_column, "timestamp", "longitude", "latitude"]]
                data_v = data_v.set_index('timestamp').resample('1h').mean().reset_index()
                data_v["all_nan"] = np.isnan(data_v[value_column]).astype(bool)
                data_v["prev_nan"] = data_v["all_nan"].shift(1).fillna(False)
                data_v["next_nan"] = data_v["all_nan"].shift(-1).fillna(False)
                data_v = data_v[~(data_v["all_nan"] & data_v["prev_nan"] & data_v["next_nan"])]

                dfs.append(pd.DataFrame({
                    "timestamp": data_v["timestamp"],
                    "latitude": data_v["latitude"],
                    "longitude": data_v["longitude"],
                    "species": value_column,
                    "value": data_v[value_column],
                    "device_id": str(self.dtr.device_id),
                    "timestamp": data_v["timestamp"],
                    "unit": unit
                }))

            df = pd.concat(dfs)

        else:
            df = DeviceTimeRange.empty_result_df()

        self.postprocess(df)


        return df

    def results_as_dataframe(self):
        data = self.get_data_from_sql()
        return self.convert_for_hourly_display(data)

    def get_species(self):
        return self.value_columns


class HourlyV3(CustomDeviceLabelMixin, CustomLocationMixin, RawDataSQLLoader):
    """
    No calibration
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_columns = ['temp', 'humidity', 'bat_v',
                              'm_no2', 'no2_w', 'no2_a']

    def table_name(self):
        return RawReading._meta.db_table

    def gps_column_names(self):
        return ["latitude", "longitude", "gps_fix"]


class HourlyV4(CustomDeviceLabelMixin, CustomLocationMixin, RawDataSQLLoader):
    """
    No calibration
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.value_columns = [
            field.name
            for field in RawReadingV4._meta.fields
            if field.name not in {'ver', 'id', 'timestamp', 'time'}
        ]
        self.sql_columns = self.value_columns

    def table_name(self):
        return RawReadingV4._meta.db_table

    def gps_column_names(self):
        return ["lat", "long", "fix"]


class HourlyV4Filtered(HourlyV4):
    """
    No calibration
    """

    def get_data_from_sql(self):
        lat_col, lon_col, gps_fix_col = self.gps_column_names()

        lat_sql = "(CASE WHEN %s = 1 THEN %s ELSE NULL END) AS latitude" % (
            gps_fix_col, lat_col
        )

        lon_sql = "(CASE WHEN %s = 1 THEN %s ELSE NULL END) AS longitude" % (
            gps_fix_col, lon_col
        )

        conditions = self.sql_conditions_as_where_clause()

        value_columns_sql = ",\n    ".join(["(%s) AS %s" % (col, col)
                                            for col in self.sql_columns])

        query = """
        SELECT
            timestamp,
            id,
            %s,
            %s,
            %s
        FROM %s %s        
        ORDER by timestamp DESC
        
        """ % (lat_sql, lon_sql, value_columns_sql, self.table_name(), conditions)

        print(query)
        data = pd.read_sql(query, connection)

        # This is to prevent pandas dropping columns of all None during resampling
        for col in self.sql_columns + ["latitude", "longitude"]:
            data[col] = data[col].astype(np.float32)

        #        if self.fill_gaps:
        print(data.shape)

        return data

    def filter(self, df, thresh=7):

        if df.shape[0] <= 1:
            return df

        def mad_based_outlier(points, thresh=3.5):
            if len(points.shape) == 1:
                points = points[:, None]
            median = np.median(points, axis=0)
            diff = np.sum((points - median) ** 2, axis=-1)
            diff = np.sqrt(diff)
            med_abs_deviation = np.median(diff)

            modified_z_score = 0.6745 * diff / med_abs_deviation

            return modified_z_score > thresh

        df_numeric = df.select_dtypes(include=[np.number])
        cols = set(df_numeric.columns)
        cols = cols.difference(set(['time', 'lat', 'long', 'latitude', 'longitude',
                                    'fix', 'speed', 'alt', 'head', 'rpm', 'id', 'ver', 'index', 'vol']))
        df_no_na = df.dropna(subset=cols)
        id_list = df_no_na.id.unique()
        id_list.sort()
        for col in cols:
            for device_id in id_list:
                df_device = df_no_na[df_no_na.id == device_id]
                data_col_id = df_device[col].values
                outliers = mad_based_outlier(data_col_id, thresh=thresh)
                n_entries = len(data_col_id)
                perc_outliers = sum(outliers) / n_entries
                if perc_outliers > 0.05:
                    print("Column: %s Device id %i Number of entries %i Share of outliers %f" % (
                        col, device_id, n_entries, perc_outliers))
                df_no_na.loc[df_device.index, col + '_outlier'] = outliers
        df_clean = df_no_na.copy()
        for col in cols:
            outlier_col = col + '_outlier'
            if outlier_col in df_clean:
                df_clean = df_clean[~df_clean[outlier_col]]

        if len(df_no_na) > 0:
            print("Total Number of entries %i Perc remaining: %f" % (len(df_no_na), len(df_clean) / len(df_no_na)))

#        df_clean[NO2_SPECIES_COLUMN] = df_clean[NO2_SPECIES_COLUMN].clip(0, 1000)

        return df_clean[df.columns]

    def results_as_dataframe(self):

        data = self.get_data_from_sql()
        data_filtered = self.filter(data)
        return self.convert_for_hourly_display(data_filtered)


class UncalibratedDeviceError(ValueError):
    pass


class HourlyV4Calibrated(HourlyV4Filtered):
    """
    Apply calibration model and return calibrated NO2
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sql_columns = [
            field.name
            for field in RawReadingV4._meta.fields
            if field.name not in {'ver', 'id', 'timestamp', 'time'}
        ]
        self.value_columns = [NO2_SPECIES_COLUMN, PM10_SPECIES_COLUMN_IN_DASHBOARD]
        self.no2_x_cols = [
            'afewrk1', 'afeaux1', 'afewrk2', 'afeaux2', 'afewrk3', 'afeaux3',
            'afept1k', 'isbwrk', 'isbaux', 'mics1', 'mics2', 'pm1tmp',
            'pm2tmp', 'pm1hum', 'pm2hum',
        ]
        self.pm10_x_cols = ['pm110a', 'pm210a', 'pm1hum', 'pm2hum', ]

        self.no2_model = None
        self.pm10_model = None

        self.no2_models_pickle = kwargs.get("no2_models_pickle", None)
        self.pm10_models_pickle = kwargs.get("pm10_models_pickle", None)
        if os.path.exists(os.path.realpath(self.no2_models_pickle)):
            with open(os.path.realpath(self.no2_models_pickle), 'rb') as f:
                from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
                all_no2_models = pickle.load(f)

                if self.dtr.device_id in all_no2_models:
                    self.no2_model = all_no2_models[self.dtr.device_id]

        if os.path.exists(os.path.realpath(self.pm10_models_pickle)):
            with open(os.path.realpath(self.pm10_models_pickle), 'rb') as f:
                from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
                all_pm10_models = pickle.load(f)

                if self.dtr.device_id in all_pm10_models:
                    self.pm10_model = all_pm10_models[self.dtr.device_id]

        if self.no2_model is None and self.no2_model is None:
            raise UncalibratedDeviceError()

        # def get_trained_pipeline(output_pickle_path):
        #     if os.path.exists(os.path.realpath(output_pickle_path)):
        #         with open(os.path.realpath(output_pickle_path), 'rb') as f:
        #             pipeline = dill.load(f)
        #             return pipeline
        # NO2
        # self.trained_pipelines["NO2"] = get_trained_pipeline(model_name)
        # if x_columns:
        #     self.x_columns = x_columns
        # else:
        #     self.x_columns = ['afewrk1', 'afeaux1', 'afewrk2', 'afeaux2', 'afewrk3', 'afeaux3', 'afept1k', 'isbwrk',
        #                       'isbaux', 'mics1', 'mics2', 'pm1tmp', 'pm2tmp', 'pm1hum', 'pm2hum', ]

    def predict_no2_one_sensor(self, data, model):
        x_cols = list(self.no2_x_cols)

        # x_cols.append("timestamp")
        # data = data[x_cols + ["timestamp"]]

        data = data.dropna(subset=x_cols)

        lags = [1, ]
        cols_to_lag = ['afewrk3', ]

        for col in cols_to_lag:
            for i in lags:
                new_col = "lag_{}_{}".format(col, i)
                data[new_col] = data[col].shift(i)
                x_cols.append(new_col)

        drop_cols = []
        for col in drop_cols:
            data.pop(col)

        # extra nans appear after lagging
        data = data.dropna(subset=x_cols)
        # data.set_index("timestamp")

        if data.shape[0] > 0:
            data[NO2_SPECIES_COLUMN] = np.exp(model.predict(data[x_cols]).reshape(len(data)))
        else:
            data[NO2_SPECIES_COLUMN] = 0 # model.predict fails on empty data

        return data

    def predict_pm10_one_sensor(self, data, model):
        x_cols = self.pm10_x_cols
        model_cols = ['pm110a', 'pm210a', 'pm1hum', 'pm2hum', 'log_pm110a', 'log_pm210a',
                      'lag_log_pm110a_1', 'lag_log_pm110a_2', 'lag_log_pm210a_1',
                      'lag_log_pm210a_2']
        # x_cols.append("timestamp")
        # data = data[x_cols + ["timestamp"]]

        data = data.dropna(subset=x_cols)

        data = data[(data.pm110a > 0) & (data.pm210a > 0)]

        data['log_pm110a'] = np.log(data.pm110a)
        data['log_pm210a'] = np.log(data.pm210a)

        lags = [1, 2]
        cols_to_lag = ['log_pm110a', 'log_pm210a']

        for col in cols_to_lag:
            for i in lags:
                new_col = "lag_{}_{}".format(col, i)
                data[new_col] = data[col].shift(i)

        drop_cols = []
        for col in drop_cols:
            data.pop(col)

        # extra nans appear after lagging
        data = data.dropna(subset=model_cols)
        # data.set_index("timestamp")

        if data.shape[0] > 0:
            data[PM10_SPECIES_COLUMN_IN_DASHBOARD] = np.exp(model.predict(data[model_cols]).reshape(len(data)))
        else:
            data[PM10_SPECIES_COLUMN_IN_DASHBOARD] = 0 # model.predict fails on empty data

        return data

    def convert(self, df):

        return df

    def results_as_dataframe(self):
        data = self.get_data_from_sql()

        if self.no2_model is not None:
            data_filtered = self.filter(data[self.no2_x_cols+['id']])
            df_no2 = self.predict_no2_one_sensor(data_filtered, self.no2_model)

            data = data.assign(
                **{NO2_SPECIES_COLUMN: df_no2[NO2_SPECIES_COLUMN]}
            )
        else:
            data[NO2_SPECIES_COLUMN] = np.nan

        if self.pm10_model is not None:
            data_filtered = self.filter(data[self.pm10_x_cols+['id']])
            df_pm10 = self.predict_pm10_one_sensor(data_filtered, self.pm10_model)
            data = data.assign(
                **{PM10_SPECIES_COLUMN_IN_DASHBOARD: df_pm10[PM10_SPECIES_COLUMN_IN_DASHBOARD]}
            )
        else:
            data[PM10_SPECIES_COLUMN_IN_DASHBOARD] = np.nan

        return self.convert_for_hourly_display(data, unit='μg/m³')

    def ttl(self):
        60


class V4Status(CustomDeviceLabelMixin, CustomLocationMixin):
    """
    No calibration
    """

    def __init__(self, _, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def table_name(self):
        return RawReadingV4._meta.db_table

    def results_as_dataframe(self):

            query = """
            SELECT
                timestamp,
                lat AS latitude,
                long AS longitude,
                id AS device_id
            FROM %s
            ORDER BY timestamp DESC
            LIMIT 300
            """ % (self.table_name())

            print(query)

            data = pd.read_sql(query, connection)
            data['species'] = np.nan
            data['species'] = 'device_id'
            data['value'] = data['device_id']

            # This is to prevent pandas dropping columns of all None during resampling
    #        for col in self.value_columns + ["latitude", "longitude"]:
    #            data[col] = data[col].astype(np.float32)

            #        if self.fill_gaps:
            return data

    def get_species(self):
        return ['device_id']

    def get_device_ids(self):
        return self.results_as_dataframe()['device_id'].drop_duplicates().tolist()

    def get_dummy_locations(self):
        """
        DataFrame of latitude and longitude indexed by device id
        """
        return pd.DataFrame({
            "device_id": self.get_device_ids(),
            "latitude": np.nan,
            "longitude": np.nan
        }).set_index('device_id')


class KCLLoader(AbstractLoader):
    """
    Load data from KCL
    """

    def __init__(self, device_time_range, *args, site_code, **kwargs):
        super().__init__(device_time_range, *args, **kwargs)
        self.dtr = device_time_range
        self.site_code = site_code

    def results_as_dataframe(self):
        query = """
SELECT
    s.latitude,
    s.longitude,
    r.value,
    r.timestamp,
    species_code AS species,
    s.site_code AS device_id
FROM thingapi_laqnreading AS r
INNER JOIN thingapi_laqnsite AS s ON s.site_code = r.site_code
WHERE r.site_code = '%s'
  AND species_code = '%s'""" % (self.site_code, self.dtr.species.upper())

        print(query)

        df = pd.read_sql(query, connection)
        df["species"] = df["species"].str.lower()
        df["unit"] = {
            'no2': 'ug/m3',
            'pm1': 'ppb',
            'pm10': 'ppbv',
            'pm25': 'ppbv'}.get('species', '')

        return df

    def get_device_ids(self):
        return [self.site_code]


class Precomputed(AbstractLoader):
    """
    This loader just caches the results of other loaders
    """

    @classmethod
    def empty_result_df_feather(self):
        radf = DeviceTimeRange.empty_result_df()
        return self.df_to_feather_bytes(radf)

    @classmethod
    def df_to_feather_bytes(self, radf):
        radf_buf = io.BytesIO()
        pyarrow.feather.write_feather(radf, radf_buf)
        return radf_buf.getvalue()

    def __init__(self, device_time_range, subloader_name, subloader_args={}):
        super().__init__(device_time_range)
        self.device_time_range = device_time_range
        self.subloader_name = subloader_name
        self.args = subloader_args.get("args", [])
        self.kwargs = subloader_args.get("kwargs", {})

    def ensure_loaded(self):
        try:
            self.storage
        except AttributeError:
            self.storage = (
                PrecomputedLoaderData
                .objects
                .filter(device_time_range=self.device_time_range)
                .order_by('id')
                .first()
            )

    def results_as_dataframe(self):
        self.ensure_loaded()
        # The result of some delving...:
        try:
            return pyarrow.feather.read_feather(io.BytesIO(self.storage.results_as_dataframe.tobytes()))
        except Exception as e:
            print(self.device_time_range.id)
            if self.storage is None:
                print("Storage was None")
            else:
                print('Data follows:')
                print(self.storage.results_as_dataframe.tobytes())
            print(e)
            return DeviceTimeRange.empty_result_df()

    def get_device_ids(self):
        self.ensure_loaded()
        if self.storage is None:
            return []
        else:
            return json.loads(self.storage.device_ids_as_json)

    def get_species(self):
        self.ensure_loaded()
        if self.storage is None:
            return []
        else:
            return json.loads(self.storage.species_as_json)

    def postprocess(self, df):
        pass

    def get_dummy_locations(self):
        self.ensure_loaded()
        if self.storage is not None:
            return pyarrow.feather.read_feather(io.BytesIO(self.storage.dummy_locations.tobytes()))
        else:
            return super().get_dummy_locations()

    @classmethod
    def precompute_all(self):
        dtrs = DeviceTimeRange.objects.filter(
            calibration_model_loader="thingapi.loaders.Precomputed"
        ).all()

        for dtr in dtrs:
            loader = dtr.instantiate_model()

            try:
                loader.precompute
            except AttributeError:
                continue

            try:
                loader.precompute()
            except Exception as e:
                traceback.print_exc()

    def subloader(self):
        return instantiate_by_name_with_args(
            self.subloader_name,
            self.device_time_range, *self.args, **self.kwargs
        )

    def precompute(self):
        storage = (
            PrecomputedLoaderData
            .objects
            .filter(device_time_range=self.device_time_range)
            .order_by('id')
            .first()
        )

        if not storage:
            storage = PrecomputedLoaderData(
                device_time_range=self.device_time_range
            )

        subloader = self.subloader()

        radf = subloader.results_as_dataframe()
        storage.results_as_dataframe = type(self).df_to_feather_bytes(radf)

        storage.dummy_locations = type(self).df_to_feather_bytes(subloader.get_dummy_locations())

        storage.device_ids_as_json = json.dumps(subloader.get_device_ids())
        storage.species_as_json = json.dumps(subloader.get_species())

        storage.updated_at = datetime.now()

        storage.save()

        PrecomputedLoaderData.objects.filter(
            ~Q(id=storage.id) &
            Q(device_time_range_id=self.device_time_range.id)
        ).all().delete()


class AbstractWeatherLoader(ABC):
    """
    This class is responsible for loading and running calibration models.
    """

    @abstractmethod
    def site_list(self):
        """
        This should return a list of sites
        """
        pass

    @abstractmethod
    def site_weather_dataframe(self, site):
        """
        This should return a dataframe of site weather.
        Columns TBC
        """
        pass

    @abstractmethod
    def direction_bins(self):
        """
        Directions bins
        """
        pass


class METWeatherLoader(AbstractWeatherLoader):

    def site_list(self):
        return [
            tup[0]
            for tup in METSite.objects.extra(where=[
                """EXISTS(
SELECT 1 FROM thingapi_metreading mr WHERE mr.site_id = thingapi_metsite.id
            )"""
            ]).values_list('name')
        ]

    def direction_bins(self):
        labels = np.linspace(0, 360, 16, endpoint=False)
        return (labels - 360 / 16 / 2, labels)

    def site_weather_raw_data(self, site_name):
        query = """
SELECT
    r.*,
    f.name AS field,
    f.unit AS unit,
    s.name AS site_name,
    s.latitude AS latitude,
    s.longitude AS longitude,
    p.start_time AS start_time,
    p.duration AS duration
FROM thingapi_metreading r
INNER JOIN thingapi_metfield f ON r.field_id = f.id
INNER JOIN thingapi_metperiod p ON r.period_id = p.id
INNER JOIN thingapi_metsite s ON r.site_id = s.id
WHERE s.name = %s
""" % adapt(site_name)
        return pd.read_sql(query, connection)

    def site_weather_dataframe(self, site_name):
        df = self.site_weather_raw_data(site_name).pivot(
            index='start_time',
            columns='field',
            values='value'
        )

        dir_bins, dir_labels = self.direction_bins()

        dir_translations = pd.Series(dir_labels, index=[
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
        ])

        df['Wind Direction'] = df['Wind Direction'].map(dir_translations)

        df["Wind Speed"] = df["Wind Speed"].astype(np.float32) * miles_to_knots
        df = df.rename(columns={"Wind Speed": "WindSpd",
                                "Wind Direction": "WindDir"})

        return df


class DummyWeatherLoader(ABC):
    """
    Dummy weather loader. Wind does a lap of the compass every 24h
    Bound to pick something up due to diurnal variation in pollutants
    """

    def site_list(self):
        return ["24h rotating wind"]

    def site_weather_dataframe(self, site):
        """
        This should return a dataframe of site weather.
        Columns TBC
        """
        start_time = '2016-01-01'
        time = pd.date_range(start_time, end=pd.datetime.now(), freq='H')
        time_days = (time - time[0]).total_seconds() / 86400
        angle = (360 * time_days) % 360

        return pd.DataFrame({
            "WindSpd": 15,
            "WindDir": angle
        }, index=time)

    def direction_bins(self):
        labels = np.linspace(0, 360, 16, endpoint=False)
        return (labels - 360 / 16 / 2, labels)


class ExperimentProxy(AbstractLoader):
    """
    This class is responsible for loading and running calibration models.
    """

    def __init__(self, device_time_range, *args, experiment_ids=[]):
        self.experiment_ids = experiment_ids
        self.dtr = device_time_range

    def __sub_dtrs__(self):
        if not hasattr(self, "__cached_dtrs__"):
            self.__cached_dtrs__ = DeviceTimeRange.objects.filter(experiment_id__in=self.experiment_ids).all()
        return self.__cached_dtrs__

    def cleanup_db_conns(self, result):
        connection.close()
        return result


    def results_as_dataframe(self):
        executor = ThreadPoolExecutor(max_workers=30)

        loaders = [
            dtr.instantiate_model()
            for dtr in self.__sub_dtrs__()
        ]

        calibrated_results_list = [DeviceTimeRange.empty_result_df()]

        for df in executor.map(lambda loader: self.cleanup_db_conns(loader.results_as_dataframe()),
                               loaders):
            calibrated_results_list.append(df)

        return pd.concat(calibrated_results_list)

    def get_device_ids(self):
        return self.results_as_dataframe()['device_id'].drop_duplicates().tolist()

    def get_species(self):
        return self.results_as_dataframe()['species'].drop_duplicates().tolist()

    def postprocess(self, df):
        pass

    def get_dummy_locations(self):
        """
        DataFrame of latitude and longitude indexed by device id
        """
        return pd.DataFrame({
            "device_id": self.get_device_ids(),
            "latitude": np.nan,
            "longitude": np.nan
        }).set_index('device_id')


HourlyDemo = HourlyV3
