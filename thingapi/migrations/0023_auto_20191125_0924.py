# Generated by Django 2.1.1 on 2019-11-25 09:24

import datetime
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('thingapi', '0022_auto_20190828_0445'),
    ]

    operations = [
        migrations.CreateModel(
            name='Alert',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', models.DateTimeField(default=datetime.datetime.now)),
                ('recipients_json', models.TextField(default='[]')),
                ('condition_class', models.CharField(max_length=256)),
                ('condition_args', models.TextField(default='{"args": [], "kwargs": {}}')),
                ('last_changed', models.DateTimeField(blank=True, default=None, null=True)),
                ('status', models.BooleanField(default=False)),
                ('experiment', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='alerts', to='thingapi.Experiment')),
            ],
        ),
    ]
