# Generated by Django 2.1.1 on 2019-08-28 04:45

import datetime
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import thingapi.models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('thingapi', '0021_auto_20190127_1419'),
    ]

    operations = [
        migrations.CreateModel(
            name='ApiKey',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', models.DateTimeField(default=datetime.datetime.now)),
                ('last_used', models.DateTimeField(default=None, null=True)),
                ('key', models.CharField(default=thingapi.models.generate_api_key, max_length=128)),
                ('client', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='thingapi.Client')),
                ('user_created', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
