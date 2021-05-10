"""
WSGI config for thingapi project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.10/howto/deployment/wsgi/
"""

import os
import site

#site.addsitedir('/v/local/lib/python2.7/site-packages')


from django.core.wsgi import get_wsgi_application

#os.environ["DJANGO_SETTINGS_MODULE"] = "thingapi.deploy_settings"

#activate_env=os.path.expanduser("/v/bin/activate_this.py")
#exec(open(activate_env).read(), dict(__file__=activate_env))

application = get_wsgi_application()
