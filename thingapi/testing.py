from abc import ABC, abstractmethod
from thingapi.models import Experiment
from thingapi.loaders import AbstractLoader
import pandas as pd


class LoaderTestCaseBase(ABC):

    def setUp(self):
        self.experiment = Experiment.objects.create()
        super().setUp()

    @abstractmethod
    def get_loader(self):
        pass

    def get_empty_loader(self):
        raise NotImplementedError

    def test_results_as_dataframe(self):
        loader = self.get_loader()

        df = loader.results_as_dataframe()
        self.assert_results_as_dataframe_valid(df)
        self.assertNotEqual(df.shape[0], 0)

    def test_results_as_dataframe_empty(self):
        try:
            loader = self.get_empty_loader()
        except NotImplementedError as e:
            return

        df = loader.results_as_dataframe()
        self.assert_results_as_dataframe_valid(df)
        self.assertEqual(df.shape[0], 0)

    def assert_results_as_dataframe_valid(self, df):
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(set(df.columns.tolist()), set(['latitude', 'longitude', 'value', 'timestamp', 'species', 'device_id', 'unit']))


class DummyLoader(AbstractLoader):

    def __init__(self, dtr, *args, **kwargs):
        self.dtr = dtr

    def results_as_dataframe(self):
        return pd.DataFrame([{
            'latitude': 51.4,
            'longitude': 0,
            'value': 12,
            'timestamp': pd.Timestamp('2018-01-01 10:00:00'),
            'species': 'NO2',
            'device_id': 13,
            'unit': 'm/s'
        }])

