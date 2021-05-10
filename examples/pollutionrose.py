import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot


# TODO: add

def get_pollution_colour(val, palette=None, n_colors=4, species='no2'):
    if palette is None:
        palette = seaborn.color_palette('Reds', n_colors=n_colors)
    if species == 'no2':
        if val <= 40:
            return palette[0]
        if val < 80:
            return palette[1]
        if val < 100:
            return palette[1]

        return palette[2]




def fix_angle_range(series):
    return ((series + 7.5) % 360) - 7.5



def get_grouped_data(df):
    spd_bins = np.arange(0, 20, 1)
    spd_labels = spd_bins[0:-1]

    dir_bins = np.arange(-7.5, 370, 15)
    dir_labels = dir_bins[0:-1]

    return df.assign(WindDir_bins=lambda df: pd.cut(fix_angle_range(df['WindDir']), bins=dir_bins, labels=dir_labels,
                                             right=False)).assign(
        WindSpd_bins=lambda df: pd.cut(df['WindSpd'], bins=spd_bins, labels=spd_labels, right=False)).replace(
        {'WindDir_bins': {360: 0}}).groupby(by=['WindSpd_bins', 'WindDir_bins'])

def get_wind_rose_data(df):
    grouped = get_grouped_data(df)
    g = grouped.count()['WindDir']
    return g.reset_index()


def get_pollution_rose_data(df, pollutant_col='no2'):
    grouped = get_grouped_data(df)
    g = grouped.mean()[pollutant_col]
    return g.reset_index()

dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

def get_wind_rose_graph(df):
    import holoviews as hv
    hv.extension("bokeh")

    g = get_wind_rose_data(df)
    return hv.HeatMap(zip(g.WindDir_bins, g.WindSpd_bins, g.WindDir)).options(radial=True, xticks=dirs, xmarks=len(dirs),start_angle=np.pi * 5 / 8, colorbar=True


def get_pollution_rose_graph(df, pollutant_col='no2'):
    import holoviews as hv
    hv.extension("bokeh")
    # TODO: make colors different depending on pollutant
    # but ok even for now
    g = get_pollution_rose_data(df, pollutant_col=pollutant_col)
    return hv.HeatMap(zip(g.WindDir_bins, g.WindSpd_bins, g.no2)).options(radial=True, xticks=dirs, xmarks=len(dirs),
                                                                   start_angle=np.pi * 5 / 8, colorbar=True)

def get_pollution_rose_graph_from_laqn(df_wind, df_laqn, pollutant_col='NO2'):
    df_joined = df_wind.join(df_laqn[df_laqn.species_code == pollutant_col], how='inner')
    df_joined[pollutant_col] = df_joined.value
    get_pollution_rose_graph(df_joined)



# OLd code with matplotlib

# directions = np.arange(0, 360, 15)
# def get_pollution_rose_data(df, pollutant_col='n02', ):
#     def speed_labels(bins, units):
#         labels = []
#         for left, right in zip(bins[:-1], bins[1:]):
#             if left == bins[0]:
#                 labels.append('calm'.format(right))
#             elif np.isinf(right):
#                 labels.append('>{} {}'.format(left, units))
#             else:
#                 labels.append('{} - {} {}'.format(left, right, units))
#
#         return list(labels)
#
#     def fix_angle_range(series):
#         return ((series+7.5) % 360) - 7.5
#
#     spd_bins = [-1, 0, 5, 10, 15, 20, 25, 30, np.inf]
#     spd_labels = speed_labels(spd_bins, units='knots')
#
#     dir_bins = np.arange(-7.5, 360, 15)
#     dir_labels = dir_bins[0:-1]
#
#     total_count = df.shape[0]
#     calm_count = df.query("WindSpd == 0").shape[0]
#
#     # wind data
#     wind_rose = (
#         df.assign(WindSpd_bins=lambda df:
#         pd.cut(df['WindSpd'], bins=spd_bins, labels=spd_labels, right=True)
#                   )
#             .assign(WindDir_bins=lambda df:
#         pd.cut(fix_angle_range(df['WindDir']), bins=dir_bins, labels=dir_labels, right=False)
#                     )
#             .replace({'WindDir_bins': {360: 0}})
#             .groupby(by=['WindSpd_bins', 'WindDir_bins'])
#             .count()['Date']
#             .unstack(level='WindSpd_bins')
#             .assign(calm=lambda df: calm_count / df.shape[0])
#             .fillna(0)
#             .sort_index(axis=1)
#             .applymap(lambda x: x / total_count * 100)
#
#     )
#
# #    for i in directions:
# #        if i not in wind_rose.index:
# #            wind_rose = wind_rose.reindex(wind_rose.index.values.tolist() + [i])
#
#     wind_rose = wind_rose.sort_index().fillna(0)
#     wind_rose['calm'] = wind_rose['calm'].sum() / len(wind_rose)
#
#     # pollution data
#     calm_pollutant_count = df[df["WindSpd"] == 0][pollutant_col].mean()
#
#     pollutant_rose = (
#         df.assign(WindSpd_bins=lambda df:
#         pd.cut(df['WindSpd'], bins=spd_bins, labels=spd_labels, right=True)
#                   )
#             .assign(WindDir_bins=lambda df:
#         pd.cut(fix_angle_range(df['WindDir']), bins=dir_bins, labels=dir_labels, right=False)
#                     )
#             .replace({'WindDir_bins': {360: 0}})
#             .groupby(by=['WindSpd_bins', 'WindDir_bins'])
#             .mean()[pollutant_col]
#             .unstack(level='WindSpd_bins')
#             .fillna(0)
#
#         # .applymap(lambda x: x / total_count * 100)
#     )
#
#     #pollutant_rose.values[0] = pollutant_rose.values[0] + pollutant_rose.values[-1]
#     #pollutant_rose.drop(labels=[360.0], axis=0, inplace=True)
#     pollutant_rose = pollutant_rose.assign(calm=lambda df: calm_pollutant_count / df.shape[0]).sort_index(axis=1)
#
#     # Calm is basically a single point
#     pollutant_rose['calm'] = pollutant_rose['calm'].sum()
#
#     return wind_rose, pollutant_rose
#
#
# def pollution_rose_figure(wind_rosedata, pollutant_rose, wind_dirs, get_pollution_colour=get_pollution_colour):
#     def _convert_dir(directions, N=None):
#         if N is None:
#             N = directions.shape[0]
#         barDir = directions * np.pi / 180. - np.pi / N
#         barWidth = 2 * np.pi / N
#         return barDir, barWidth
#
#     bar_dir, bar_width = _convert_dir(wind_dirs)
#
#     fig, ax = pyplot.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
#     ax.set_theta_direction('clockwise')
#     ax.set_theta_zero_location('N')
#
#     for n, (c1, c2) in enumerate(zip(wind_rosedata.columns[:-1], wind_rosedata.columns[1:])):
#         if n == 0:
#             # first column only, the calm one
#             ax.bar(bar_dir, wind_rosedata[c1].values,
#                    width=bar_width,
#                    color=pollutant_rose[c1].apply(get_pollution_colour),
#                    edgecolor='none',
#                    label=c1,
#                    linewidth=0)
#
#         # all other columns
#         ax.bar(bar_dir, wind_rosedata[c2].values,
#                width=bar_width,
#                bottom=wind_rosedata.cumsum(axis=1)[c1].values,
#                color=pollutant_rose[c2].apply(get_pollution_colour),
#                edgecolor='none',
#                label=c2,
#                linewidth=0)
#
#     # leg = ax.legend(loc=(0.75, 0.95), ncol=2)
#     xtl = ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
#     ytl = ax.set_yticks(np.arange(0, 60, 10))
#
#     return fig
#
#
# def get_pollution_rose_figure(csv_path, pollutant_col='no2', n_pollution_colors=4):
#     df = pd.read_csv(csv_path)
#     wind_rosedata, pollutant_rose = get_pollution_rose_data(df, pollutant_col)
#     return pollution_rose_figure(wind_rosedata, pollutant_rose, directions,
#                                  lambda x: get_pollution_colour(x, None, n_pollution_colors, pollutant_col))
