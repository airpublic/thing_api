# -*- coding: utf-8 -*-
# Generated by Django 1.10.3 on 2016-12-10 01:02
from __future__ import unicode_literals

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('thingapi', '0002_auto_20161209_1826'),
    ]

    operations = [
        migrations.AlterField(
            model_name='organicitytoken',
            name='access_token',
            field=models.CharField(max_length=2048),
        ),
        migrations.AlterField(
            model_name='organicitytoken',
            name='refresh_token',
            field=models.CharField(max_length=2048),
        ),
        migrations.AlterField(
            model_name='reading',
            name='last_organicity_sync',
            field=models.DateTimeField(blank=True, null=True),
        )
    ]
