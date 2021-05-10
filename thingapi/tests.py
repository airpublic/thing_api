from django.test import TestCase, Client
from thingapi.models import (
    Reading, RawReading, RawReadingV4, DeviceTimeRange, Experiment,
    PrecomputedLoaderData
)
from thingapi.loaders import (
    HourlyV3, ExperimentProxy, AbstractLoader, Precomputed
)
from collections import OrderedDict
from abc import ABC, abstractmethod
import pandas as pd
import json
from thingapi.testing import LoaderTestCaseBase


class ArduinoTestCase(TestCase):

    def test_sparkfun_compatability(self):

        tup = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0
               )

        url = "/v1/readings/"
        data = "co=%s&latitude=%s&longitude=%s&no2=%s&pm1=%s&pm2_5=%s&pm10=%s&sampling_period=%s&temp=%s&humidity=%s&date=%s&gps_fix=%u&altitude=%s&speed=%s&heading=%s&lonet_bat=%u&sample_flow_rate=%s&gas_temp=%s&no2_aux=%s&no2_working=%s&co_aux=%s&co_working=%s&id=0&retries=0" % tup

        client = Client()

        before_count = Reading.objects.count()

        response = client.post(url, data, content_type='application/x-www-form-urlencoded')

        after_count = Reading.objects.count()

        self.assertEqual(after_count, before_count + 1)

        print(response.data)


class ArduinoRawReadingsTestCase(TestCase):

    def test_sparkfun_compatability(self):

        tup = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0
               )

        url = "/v2/readings/"
        data = "location_std=%s&bat_v=%s&unexpected_field=123.4&latitude=%s&longitude=%s&pm1=%s&pm2_5=%s&pm10=%s&sampling_period=%s&temp=%s&humidity=%s&date=%s&gps_fix=%u&altitude=%s&speed=%s&heading=%s&lonet_bat=%u&sample_flow_rate=%s&gas_temp=%s&no2_a=%s&no2_w=%s&co_a=%s&co_w=%s&id=0&retries=0" % tup

        client = Client()

        before_count = RawReading.objects.count()

        response = client.post(url, data, content_type='application/x-www-form-urlencoded')

        after_count = RawReading.objects.count()

        self.assertEqual(after_count, before_count + 1)

        print(response.data)


class ReadingsV4TestCase(TestCase):

    def test_sparkfun_compatability(self):

        fields = OrderedDict(
            id=[8, int, 8, None],
            Ver=[4, int, 8, None],
            Rtys=[4, int, 8, None],
            vol=[8, int, 16, None],
            afewrk1=[16, int, 16, None],
            afeaux1=[16, int, 16, None],
            afewrk2=[16, int, 16, None],
            afeaux2=[16, int, 16, None],
            afewrk3=[16, int, 16, None],
            afeaux3=[16, int, 16, None],
            afept1k=[16, int, 16, None],
            isbwrk=[16, int, 16, None],
            isbaux=[16, int, 16, None],
            mics1=[16, int, 16, None],
            mics2=[16, int, 16, None],
            time=[0, int, 160, str],
            lat=[32, float, 32, None],
            long=[32, float, 32, None],
            alt=[0, float, 32, None],
            fix=[0, int, 8, None],
            speed=[0, float, 32, None],
            head=[0, float, 32, None],
            rpm=[8, int, 16, None],
            pm11c=[8, int, 16, None],
            pm125c=[8, int, 16, None],
            pm110c=[8, int, 16, None],
            pm11a=[0, int, 16, None],
            pm125a=[0, int, 16, None],
            pm110a=[0, int, 16, None],
            pm1par3=[0, int, 16, None],
            pm1par5=[0, int, 16, None],
            pm1par10=[0, int, 16, None],
            pm1par25=[0, int, 16, None],
            pm1tmp=[8, int, 16, None],
            pm1hum=[8, int, 16, None],
            pm21c=[8, int, 16, None],
            pm225c=[8, int, 16, None],
            pm210c=[8, int, 16, None],
            pm21a=[0, int, 16, None],
            pm225a=[0, int, 16, None],
            pm210a=[0, int, 16, None],
            pm2par3=[0, int, 16, None],
            pm2par5=[0, int, 16, None],
            pm2par10=[0, int, 16, None],
            pm2par25=[0, int, 16, None],
            pm2tmp=[8, int, 16, None],
            pm2hum=[8, int, 16, None]
        )

        data = ["%s=%s" % (k, str(cons(0)))
                for k, v in fields.items()
                for ln, lt, gn, gt in [v]
                for cons in [(lt or gt)]]

        data = "&".join(data)

        url = "/v4/readings/"

        client = Client()

        before_count = RawReadingV4.objects.count()

        response = client.post(url, data, content_type='application/x-www-form-urlencoded')

        after_count = RawReadingV4.objects.count()

        self.assertEqual(after_count, before_count + 1)

        print(response.data)


class HourlyV3TestCase(LoaderTestCaseBase, TestCase):

    def setUp(self):
        super().setUp()
        if not RawReading.objects.filter(id=1234567).exists():
            r = RawReading.dummy()
            r.id = 1234567
            r.save()
        self.dtr = DeviceTimeRange.objects.create(
                device_id=1234567,
                species="N2",
                experiment_id=self.experiment.id
        )

    def get_loader(self):
        return HourlyV3(self.dtr)


class ExperimentProxyTestCase(LoaderTestCaseBase, TestCase):

    def setUp(self):
        super().setUp()
        self.sub_experiment = Experiment.objects.create()
        self.sub_dtr = DeviceTimeRange.objects.create(
            device_id=1234567,
            species="N2",
            experiment_id=self.sub_experiment.id,
            calibration_model_loader='thingapi.testing.DummyLoader',
            calibration_model_args=json.dumps({"kwargs": {}, "args": []})
        )
        self.dtr = DeviceTimeRange.objects.create(
                device_id=890,
                species="N2",
                experiment_id=self.experiment.id,
                calibration_model_args=json.dumps({"kwargs": {}, "args": []})
        )

    def get_loader(self):
        return ExperimentProxy(self.dtr, experiment_ids=[self.sub_experiment.id])


class PrecomputedLoaderMixin():

    def setUp(self):
        super().setUp()

        self.dtr = DeviceTimeRange.objects.create(
                device_id=890,
                species="N2",
                experiment_id=self.experiment.id,
                calibration_model_args=json.dumps({"kwargs": {}, "args": []})
        )

    def get_loader(self):
        return Precomputed(
            self.dtr, subloader_name="thingapi.testing.DummyLoader",
            subloader_args={"args": [], "kwargs": {}}
        )


class PrecomputedLoaderTestCase(
        PrecomputedLoaderMixin, LoaderTestCaseBase, TestCase
        ):

    # This test case runs the standard loader tests on
    # thingapi.loaders.Precomputed

    def setUp(self):
        super().setUp()

        loader = self.get_loader()
        loader.precompute()


class PrecomputedLoaderPrecomputationTestCase(
        PrecomputedLoaderMixin, TestCase):

    def setUp(self):
        self.experiment = Experiment.objects.create()
        self.irrelevant_dtr = DeviceTimeRange.objects.create(
                device_id=890,
                species="N2",
                experiment_id=self.experiment.id,
                calibration_model_args=json.dumps({"kwargs": {}, "args": []})
        )
        irrelevant_data = PrecomputedLoaderData.objects.create(
            device_time_range_id=self.irrelevant_dtr.id
        )
        super().setUp()

    # This test case checks that the precomputation actually works

    def test_precomputation_saves_data(self):

        loader = self.get_loader()
        before = PrecomputedLoaderData.objects.count()
        loader.precompute()
        after = PrecomputedLoaderData.objects.count()
        self.assertEqual(after, before+1)

    def test_precomputation_updates_data(self):
        loader = self.get_loader()
        loader.precompute()
        last_update = PrecomputedLoaderData.objects.last().updated_at

        before = PrecomputedLoaderData.objects.count()
        loader.precompute()
        after = PrecomputedLoaderData.objects.count()
        self.assertEqual(after, before)

        new_update = PrecomputedLoaderData.objects.last().updated_at
        self.assertGreater(new_update, last_update)

    def test_precomputation_deletes_duplicates(self):
        before = PrecomputedLoaderData.objects.count()

        loader = self.get_loader()
        # Save PrecomputedLoaderData 1
        loader.precompute()
        # Save PrecomputedLoaderData 2
        dupe_data = PrecomputedLoaderData.objects.create(
            device_time_range_id=self.dtr.id
        )

        # There are now 2x PrecomputedLoaderData, but afterwards there should
        # only be 1
        loader.precompute()
        after = PrecomputedLoaderData.objects.count()
        self.assertEqual(after, before)

