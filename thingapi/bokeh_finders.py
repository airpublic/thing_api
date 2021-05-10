from abc import abstractmethod, ABC
from random import randint
import os
from django.conf import settings
from urllib.parse import urlparse


class BokehFinder(ABC):

    def __init__(self, request):
        self.request = request
        self.bokeh_app_path = "/experiment"
        self.bokeh_url = '%s%s' % (self.bokeh_root, self.bokeh_app_path)
        self.bokeh_internal_url = '%s%s' % (self.bokeh_internal_root, self.bokeh_app_path)
        self.bokeh_root_with_slash = "%s/" % self.bokeh_root

    @abstractmethod
    def bokeh_root_(self):
        pass

    @abstractmethod
    def bokeh_internal_root(self):
        pass

    @property
    def bokeh_root(self):
        return self.bokeh_root_()

    @property
    def bokeh_internal_root(self):
        return self.bokeh_internal_root_()


class EnvFixedBokehFinder(BokehFinder):

    def bokeh_root_(self):
        return settings.BOKEH_ROOT

    def bokeh_internal_root_(self):
        return settings.BOKEH_INTERNAL_ROOT


class ProductionBokehFinder(BokehFinder):

    def __init__(self, request):
        self.bokeh_process_number = randint(
                0,
                int(settings.BOKEH_NUM_PROCESSES)-1
        )
        super().__init__(request)

    def bokeh_root_(self):
        hostname = urlparse(settings.BOKEH_ROOT).hostname
        return "https://bokeh-%d.%s" % (self.bokeh_process_number, hostname)

    def bokeh_internal_root_(self):
        return "http://localhost:%d" % (5000+self.bokeh_process_number)
