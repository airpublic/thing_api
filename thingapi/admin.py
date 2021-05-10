from django.contrib import admin
from django.contrib.postgres.fields import JSONField
from django.forms import ModelForm
from prettyjson import PrettyJSONWidget
from thingapi.models import (
    Reading, Client, Experiment, ClientExperiment,
    DeviceTimeRange, WeatherDataSource, ClientUser, Alert)


class DeviceTimeRangeForm(ModelForm):
    class Meta:
        model = DeviceTimeRange
        fields = '__all__'
        widgets = {
          'calibration_model_args': PrettyJSONWidget(),
        }


class AlertForm(ModelForm):
    class Meta:
        model = Alert
        fields = '__all__'
        widgets = {
          'condition_args': PrettyJSONWidget(),
        }


class ClientExperimentInline(admin.TabularInline):
    model = ClientExperiment
    extra = 1


class ClientUserInline(admin.TabularInline):
    model = ClientUser
    extra = 1


class DeviceTimeRangeInline(admin.TabularInline):
    model = DeviceTimeRange
    extra = 0
    form = DeviceTimeRangeForm


class AlertInline(admin.TabularInline):
    model = Alert
    extra = 0
    form = AlertForm


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    ordering = ['name', 'id']
#    search_fields = ['name']
#    autocomplete_fields = ['clients']
    inlines = (ClientExperimentInline, DeviceTimeRangeInline, AlertInline)


@admin.register(Client)
class ClientAdmin(admin.ModelAdmin):
    ordering = ['name', 'id']
#    search_fields = ['name']
#    autocomplete_fields = ['experiments']
    inlines = (ClientExperimentInline, ClientUserInline)




@admin.register(DeviceTimeRange)
class DeviceTimeRangeAdmin(admin.ModelAdmin):
    ordering = ['start_timestamp']
    form = DeviceTimeRangeForm


@admin.register(WeatherDataSource)
class WeatherDataSourceAdmin(admin.ModelAdmin):
    ordering = ['id']


admin.site.register(Reading)
