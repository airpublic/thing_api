from thingapi.models import Reading, RawReading, RawReadingV4, Experiment
from rest_framework import serializers
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np


class ReadingSerializer(serializers.ModelSerializer):

    class Meta:
        model = Reading
        fields = '__all__'

class RawReadingSerializer(serializers.ModelSerializer):

    class Meta:
        model = RawReading
        fields = '__all__'


class CaseInsensitiveSerializer():

    def to_internal_value(self, data):
        lowercased = {
            k.lower(): v
            for k, v in data.items()
        }

        return super().to_internal_value(lowercased)


class RawReadingV4Serializer(CaseInsensitiveSerializer,
                             serializers.ModelSerializer):

    class Meta:
        model = RawReadingV4
        fields = '__all__'


class SmallResponseSerializer:

    def to_representation(self, *args, **kwargs):

        return {}


class ReadingSerializerSmallResponse(ReadingSerializer, SmallResponseSerializer):
    pass


class RawReadingSerializerSmallResponse(RawReadingSerializer, SmallResponseSerializer):
    pass


class RawReadingV4SerializerSmallResponse(RawReadingV4Serializer,
                                          SmallResponseSerializer):
   pass


class ExperimentsV1Serializer(serializers.ModelSerializer):

    summary = serializers.SerializerMethodField()
    data = serializers.SerializerMethodField()

    class Meta:
        model = Experiment
        fields = ['id', 'name', 'summary', 'data']

    def get_device_ids(self):
        device_ids = self.context['request'].GET.get('device_ids', None)

        if device_ids:
            try:
                return device_ids.split(",")
            except Exception as e:
                raise ValidationError("device_ids was invalid")

    def get_species(self):
        species = self.context['request'].GET.get('species', None)

        if species:
            try:
                return species.split(",")
            except Exception as e:
                raise ValidationError("species was invalid")

    def get_start_time(self):
        try:
            ts = pd.to_datetime(self.context['request'].GET.get('start_timestamp', None))
            if ts:
                ts = ts.tz_localize('UTC')
            return ts
        except ValueError:
            raise ValidationError("start_timestamp was invalid")

    def get_end_time(self):
        try:
            ts = pd.to_datetime(self.context['request'].GET.get('end_timestamp', None))
            if ts:
                ts = ts.tz_localize('UTC')
            return ts
        except ValueError:
            raise ValidationError("end_timestamp was invalid")

    def get_df(self):
        try:
            return self.__df__
        except AttributeError as e:
            executor = ThreadPoolExecutor(max_workers=30)

            loaders = [
                dtr.instantiate_model()
                for dtr in self.instance.device_time_ranges.all()
            ]

            # TODO - add timeout
            calibrated_results_list = []

            for df in executor.map(lambda loader: loader.results_as_dataframe(),
                                   loaders):
                calibrated_results_list.append(df)

            self.__df__ = pd.concat(calibrated_results_list, sort=False)
            return self.__df__

    def get_filtered_df(self):
        df = self.get_df()
        device_ids = self.get_device_ids()
        species = self.get_species()
        start_time = self.get_start_time()
        end_time = self.get_end_time()

        if device_ids:
            df = df[df["device_id"].isin(device_ids)]

        if species:
            df = df[df["species"].isin(species)]

        if start_time:
            df = df[df["timestamp"] >= start_time]

        if end_time:
            df = df[df["timestamp"] < end_time]

        return df

    def get_data(self, obj):

        def extract_data_as_dict(g):
            g = g.sort_values("timestamp")
            return {
                "latitude": g["latitude"].tolist(),
                "longitude": g["longitude"].tolist(),
                "timestamp": g["timestamp"].tolist(),
                "value": g["value"].tolist(),
                "species": g["species"].iloc[0],
                "device_id": g["device_id"].iloc[0],
            }

        def replace_nans_with_nones(series):
            if series.dtype in (np.float32, np.float64):
                return series.replace({np.nan: None})
            else:
                return series

        series_series = (self.get_filtered_df()
            .apply(replace_nans_with_nones, axis=0)
            .groupby(by=['device_id', 'species'])
            .apply(extract_data_as_dict)
        )

        if isinstance(series_series, pd.Series):
            return series_series.tolist()
        elif series_series.shape[0] == 0:
            return []
        else:
            raise ValueError("Unable to convert this value to list")

    def get_summary(self, obj):
        df = self.get_df()
        return {
            "start_time": df["timestamp"].min(),
            "end_time": df["timestamp"].max(),
            "device_ids": df["device_id"].drop_duplicates().tolist(),
            "species": df["species"].drop_duplicates().tolist(),
        }
