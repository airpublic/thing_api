"""thingapi URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from thingapi.views import (RawReadingsViewSet, ReadingsViewSet, dashboard,
                            RawReadingsV4ViewSet, experiments, experiment,
                            client, api_keys, delete_api_key, root,
                            ExperimentsV1ViewSet)
from thingapi.views import devices, latest, device_status, device_status_v4

from rest_framework import routers

from django.contrib.auth import urls as django_auth_urls
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

from drf_yasg.views import get_schema_view
from drf_yasg import openapi

from thingapi import docs

router = routers.SimpleRouter()

router.register(r'v1/readings', ReadingsViewSet)
router.register(r'v2/readings', RawReadingsViewSet)
router.register(r'v4/readings', RawReadingsV4ViewSet)
router.register(r'v1/experiments', ExperimentsV1ViewSet, base_name="v1-experiment")

public_schema_view = get_schema_view(
   openapi.Info(
      title="",
      default_version='',
       description="""
## Getting Started

1. **Obtain API key**
    You can get this by logging into AirPublic, and clicking a settings icon.

2. **Make calls to our API**
    You can use this token when interfacing with out API.

""",
      terms_of_service="",
      contact=openapi.Contact(email="info@airpublic.eu")
   ),
   validators=['flex'],
   public=True,
   generator_class=docs.PublicOpenAPISchemaGenerator,
)

urlpatterns = [
    url(r'^', include(django_auth_urls)),
    url(r'^experiments/?$', experiments),
    url(r'^experiments/(?P<id>\d+)/?$', experiment),
    url(r'^clients/(?P<id>\d+)', client),
    url(r'^api_keys/?$', api_keys),
    url(r'^api_keys/(?P<id>\d+)/delete/', delete_api_key),
    url(r'^admin/', admin.site.urls),
    url(r'^v1/latest', latest),
    url(r'^v1/devices', devices),
    url(r'^', include(router.urls)),
    url(r'^v1/device_status/(?P<id>\d+)$', device_status),
    url(r'^v4/device_status/(?P<id>\d+)$', device_status_v4),
    url(r'^$', root),
    url(r'^engdash^$', dashboard),
    url(r'^reference(?P<format>\.json|\.yaml)$', public_schema_view.without_ui(cache_timeout=0), name='schema-json'),
    url(r'^reference/?$', public_schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
]

urlpatterns += staticfiles_urlpatterns()
