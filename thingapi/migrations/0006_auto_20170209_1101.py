# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-02-09 11:01
from __future__ import unicode_literals

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('thingapi', '0005_auto_20161230_1830'),
    ]

    operations = [
        migrations.AddField(
            model_name='rawreading',
            name='m_co',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='rawreading',
            name='m_no2',
            field=models.FloatField(null=True),
        )
    ]
