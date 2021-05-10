from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.http import HttpResponseForbidden
from rest_framework import viewsets
from rest_framework import mixins
from rest_framework.exceptions import NotFound, ValidationError
from thingapi.models import (
        Reading, RawReading, RawReadingV4, Experiment, Client, ApiKey)
from thingapi.serializers import (ReadingSerializerSmallResponse,
                                  RawReadingSerializerSmallResponse,
                                  RawReadingV4SerializerSmallResponse,
                                  ExperimentsV1Serializer)
import thingapi.plotting
from django.contrib.auth.decorators import login_required
import pandas as pd
from datetime import datetime
from bokeh.embed import server_document
from bokeh.client import pull_session
from bokeh.embed import server_session
import os
import logging

# The following imports are required for a nasty hack to get Bokeh to work
from bokeh.embed.server import server_html_page_for_session
from bokeh.resources import Resources
from bokeh.server.views.static_handler import StaticHandler
from jinja2.environment import Template
import bokeh.embed.elements

from django.conf import settings
from thingapi.models import instantiate_by_name_with_args
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.utils.decorators import method_decorator


class ReadingsViewSet(mixins.CreateModelMixin,
               viewsets.GenericViewSet):

    queryset = Reading.objects.all()
    serializer_class = ReadingSerializerSmallResponse


class RawReadingsViewSet(mixins.CreateModelMixin,
               viewsets.GenericViewSet):

    queryset = RawReading.objects.all()
    serializer_class = RawReadingSerializerSmallResponse


class RawReadingsV4ViewSet(mixins.CreateModelMixin,
               viewsets.GenericViewSet):

    queryset = RawReadingV4.objects.all()
    serializer_class = RawReadingV4SerializerSmallResponse

    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        if response.status_code != 201:
            logger.warning("Failed v4 POST postmortem. Status code was %d, data was '%s'" % (response.status_code, response.data))
        return response


@login_required(login_url="/login/")
def root(request):

    return redirect('/experiments', request)


@login_required(login_url="/admin/login/")
def dashboard(request):

    return redirect('/experiments', request)

#    script, div = thingapi.plotting.readings_per_day()
#    tscript, tdiv = thingapi.plotting.latest_readings()
#    mscript, mdiv = thingapi.plotting.map_plot()
#    dscript, ddiv = thingapi.plotting.device_statuses()
#
#    return render(request, "index.hamlpy", {"timeseries": script+div, "latest": tscript+tdiv, "map": mscript+mdiv, "devices": dscript+ddiv})

@login_required(login_url="/admin/login/")
def devices(request):

    data = thingapi.plotting.device_statuses_data_source().to_json(include_defaults=False)['data']
    return JsonResponse(data)

@login_required(login_url="/admin/login/")
def latest(request):

    data = thingapi.plotting.latest_readings_data_source().to_json(include_defaults=False)['data']
    return JsonResponse(data)


def device_status(request, **kwargs):

    device_id = int(kwargs["id"])

    last_reading = thingapi.models.get_device_last_reading(device_id)
    last_reading_json = last_reading.iloc[0].to_json()

    current_time = pd.to_datetime(datetime.utcnow()).tz_localize('UTC')
    last_reading_time = last_reading.iloc[0]['timestamp'].tz_convert('UTC')

    if (current_time - last_reading_time).total_seconds() < 80:
        return HttpResponse(status=200,
                            content=last_reading_json,
                            content_type='application/json')

    else:
        return HttpResponse(status=503,
                            content=last_reading_json,
                            content_type='application/json')

def device_status_v4(request, **kwargs):

    device_id = int(kwargs["id"])

    last_reading = thingapi.models.get_device_last_reading_v4(device_id)
    last_reading_json = last_reading.iloc[0].to_json()

    current_time = pd.to_datetime(datetime.utcnow()).tz_localize('UTC')
    last_reading_time = last_reading.iloc[0]['timestamp'].tz_convert('UTC')

    if (current_time - last_reading_time).total_seconds() < 80:
        return HttpResponse(status=200,
                            content=last_reading_json,
                            content_type='application/json')

    else:
        return HttpResponse(status=503,
                            content=last_reading_json,
                            content_type='application/json')


@login_required(login_url="/login/")
def experiments(request):
    clients = request.user.clients.all()
    return render(request, 'experiments.html', {
        "clients_experiments": [(c, c.experiments.order_by('id').all()) for c in clients],
    })


@login_required(login_url="/login/")
def experiment(request, id=None):

    clients = request.user.clients.all()
    experiment = (
        Experiment.objects.filter(client__in=clients).filter(id=id).first()
    )
    bokeh_finder = instantiate_by_name_with_args(
        settings.BOKEH_FINDER_CLASS, request)

    autoload = bokeh.embed.server_document(
        url='%sexperiment' % bokeh_finder.bokeh_root_with_slash,
        arguments={'experiment_id': id}
    )

    return render(request, "experiment.html", {
        "experiment": experiment,
        "bokeh_finder": bokeh_finder,
        "autoload": autoload
    })


@login_required(login_url="/login/")
def client(request, id=None):
    client_ = request.user.clients.filter(id=id).first()

    return render(request, 'client.html', {
        "client": client_,
        "api_keys": client_.api_keys.select_related('user_created').all()
    })


@login_required(login_url="/login/")
def api_keys(request):
    api_key = ApiKey(**request.POST.dict())
    api_key.user_created = request.user
    api_key.save()

    return redirect('/clients/%s/' % api_key.client_id)


@login_required(login_url="/login/")
def delete_api_key(request, id):
    api_key = ApiKey.objects.filter(id=id).first()

    if request.user.id not in {u.id for u in set(api_key.client.users.only('id'))}:
        return HttpResponseForbidden()

    client_id = api_key.client_id

    api_key.delete()

    return redirect('/clients/%s/' % client_id)


@method_decorator(
    name='retrieve',
    decorator=swagger_auto_schema(
        responses={
            200: "Success. Response contains data.",
            404: "Experiment not found in your account",
            413: "Payload too large. Please request less data"
        },
        operation_description="""
Get experiment data
===================

This endpoint retrieves data from an AirPublic experiment. The response will \
contain a summary with a list of available device ids and species, along with \
the actual data. Authentication is available through logging in, or using the \
`api_key` GET parameter. Since the response from this endpoint can become quite \
large and unwieldy, any request resulting in greater than 98304 values will \
result in a 413 status code.

""",
        manual_parameters=[
            openapi.Parameter(
                'api_key',
                openapi.IN_QUERY,
                description="Airpublic API key. You can generate these by visiting the client settings page.",
                type=openapi.TYPE_STRING
            ),
            openapi.Parameter(
                'start_timestamp',
                openapi.IN_QUERY,
                description="Only return readings from on or after this ISO 8601 timestamp.",
                type=openapi.TYPE_STRING
            ),
            openapi.Parameter(
                'end_timestamp',
                openapi.IN_QUERY,
                description="Only return readings before this ISO 8601 timestamp.",
                type=openapi.TYPE_STRING
            ),
            openapi.Parameter(
                'species',
                openapi.IN_QUERY,
                description="Only return readings from this comma separated list of species",
                type=openapi.TYPE_STRING
            ),
            openapi.Parameter(
                'device_ids',
                openapi.IN_QUERY,
                description="Only return readings from this comma separated list of device ids",
                type=openapi.TYPE_STRING
            ),
        ]
    ),
)
class ExperimentsV1ViewSet(
        mixins.RetrieveModelMixin,
        viewsets.GenericViewSet):

    serializer_class = ExperimentsV1Serializer
    queryset = Experiment.objects.all()

    def get_clients(self):
        api_key = self.request.GET.get('api_key', None)

        if api_key:
            return Client.objects.filter(api_keys__key=api_key).all()
        else:
            return self.request.user.clients.all()

    def get_object(self):
        id = self.kwargs['pk']
        experiment = (
            Experiment.objects.filter(client__in=self.get_clients()).filter(id=id).first()
        )
        if not experiment:
            raise NotFound("Experiment not found")
        return experiment
