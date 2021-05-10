# Generated by Django 2.1.1 on 2018-09-19 10:48

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('thingapi', '0010_auto_20180919_1008'),
    ]

    operations = [
        migrations.AddField(
            model_name='client',
            name='experiments',
            field=models.ManyToManyField(through='thingapi.ClientExperiment', to='thingapi.Experiment'),
        ),
        migrations.AddField(
            model_name='experiment',
            name='clients',
            field=models.ManyToManyField(through='thingapi.ClientExperiment', to='thingapi.Client'),
        )
    ]
