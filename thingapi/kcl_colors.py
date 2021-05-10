from bokeh.models.mappers import LinearColorMapper
import holoviews as hv


kcl_no2_colors = [
    "#0000FF",
    "#0078ff",
    "#0096ff",
    "#00c8ff",
    "#00dcff",
    "#00dcc8",
    "#00dc96",
    "#c8f064",
    "#ffff00",
    "#fff500",
    "#ffeb00",
    "#ffe100",
    "#ffd700",
    "#ffcd00",
    "#ffa000",
    "#ff7d00",
    "#ff6400",
    "#ff5000",
    "#ff3c00",
    "#ff2800",
    "#ff1400",
    "#ff0000",
    "#e60000",
    "#d20000",
    "#be0000",
    "#aa0000",
    "#820000",
    "#500000",
]


kcl_no2_levels = [
    0, 16, 19,22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 100
]

kcl_clipping_colors = {'min': '#000000', 'max': kcl_no2_colors[-1], 'NaN': '#CCCCCC'}
