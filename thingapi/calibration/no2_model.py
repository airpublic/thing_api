"""
In order to get calibrated air quality measurements out of the raw uncalibrated data stored in the database, see our paper https://amt.copernicus.org/preprints/amt-2020-473/. The accompanying code will be published on github by @peernow soon.

This file contains an old version of calibration which is usable but less accurate.
"""

def prepareData(data, lags=[24], test_size=0.2, cols_to_lag=[], drop_cols=[]):
    data = pd.DataFrame(data.copy())

    # считаем индекс в датафрейме, после которого начинается тестовыый отрезок
    data = data.dropna()
    test_index = int(len(data) * (1 - test_size))

    # добавляем лаги исходного ряда в качестве признаков
    for col in cols_to_lag:
        for i in lags:
            data["lag_{}_{}".format(col, i)] = data[col].shift(i)

    # data.index = data.index.to_datetime()

    #     # считаем средние только по тренировочной части, чтобы избежать лика
    #     data['weekday_average'] = data.dayofweek.map(code_mean(data[:test_index], 'dayofweek', "y").get )
    #     data["hour_average"] = data.hr.map(code_mean(data[:test_index], 'hr', "y").get )
    #     data["timeofday_average"] = data.timeofday.map(code_mean(data[:test_index], 'timeofday', "y").get )
    #     data["weekend_average"] = data.weekend.map(code_mean(data[:test_index], 'weekend', "y").get )

    #     # выкидываем закодированные средними признаки
    #     data.drop(["hr", "dayofweek", "weekend","timeofday"], axis=1, inplace=True)

    data = data.dropna()
    data = data.reset_index(drop=True)

    for col in drop_cols:
        data.pop(col)

    # разбиваем весь датасет на тренировочную и тестовую выборку
    X_train = data.loc[:test_index].drop(["y"], axis=1)
    y_train = data.loc[:test_index]["y"]
    X_test = data.loc[test_index:].drop(["y"], axis=1)
    y_test = data.loc[test_index:]["y"]

    return X_train, X_test, y_train, y_test

def fit_evaluate_pls(X_train, X_test, y_train, y_test, n=3):
    pls1 = PLSRegression(n_components=n)
    pls1.fit(X_train, y_train)
    model = pls1
    # print(model.summary())

    print(X_test.shape)
    y_hats = pls1.predict(X_test).reshape(len(X_test))

    test_resid = y_hats - y_test
    test_student_resid = test_resid / test_resid.std()
    plt.scatter(y_hats, test_student_resid, alpha=.35)

    print(r2_score(y_train, model.predict(X_train).reshape(len(y_train)), multioutput='raw_values'))
    print(r2_score(y_test, model.predict(X_test).reshape(len(y_test)), multioutput='raw_values'))
    print(len(X_train))
    len(X_test)
    return model, y_test, y_hats

def pls_model(n):
    data = df[df['kings_no2'] >= 0].copy()

    # data = data[(data['afewrk3'] < 300) & (data['afeaux3'] < 285)]

    #     x_cols =  ['co_a', 'co_w', 'no2_a', 'no2_w', 'm_co', 'm_no2', 'w_pm10', 'pm10', 'w_pm1', 'pm1', 'w_pm2_5', 'temp', 'humidity',
    #            'weekend','timeofday', 'hr', 'dayofweek'

    x_cols = ['afewrk1', 'afeaux1', 'afewrk2', 'afeaux2', 'afewrk3', 'afeaux3', 'afept1k', 'isbwrk', 'isbaux', 'mics1',
              'mics2', 'pm1tmp', 'pm2tmp', 'pm1hum', 'pm2hum', ]

    x_cols.append('timestamp')
    x_cols.append('kings_no2')

    drop_cols = []  # ['mics1', 'timeofday',  ]#    'pm11c', 'pm125c', 'pm110c', 'pm11a', 'pm125a', 'pm110a','pm1par3', 'pm1par5', 'pm1par10', 'pm1par25','pm21c', 'pm225c', 'pm210c', 'pm21a', 'pm225a', 'pm210a', 'pm2par3','pm2par5', 'pm2par10', 'pm2par25',] #
    # this gives 0.62 without mean coding['w_pm1', 'w_pm10', 'w_pm2_5', 'pm1', 'pm10', 'no2_w']

    data['log_afewrk3'] = np.log(data.afewrk3)

    data = data[x_cols]
    data['y'] = np.log(data.pop('kings_no2'))
    # data['y'] =  data.pop('kings_no2')

    # # log scale
    # data['log_no2_w'] = np.log(data.no2_w)

    data = data.set_index('timestamp')
    # data['const'] = 1

    return fit_evaluate_pls(
        *prepareData(data, lags=[1, ], cols_to_lag=['afewrk3', ], drop_cols=drop_cols, test_size=0.2), n=n)

    # yay .. got up to 0.61 on the test without mean coding
    # with mean coding it's 0.68
    # with 24*7 lag it's 0.72 BUT on a very small test set. Need 1k values to learn :()

def get_importance(model, x_cols):
    """
    variable selection for pls regression
    :param model:
    :param x_cols:
    :return:
    """
    def vip(model):
        """
         from a review of variable selection methods in partial least squares regression "a proper threshold between 0.83 and 1.21 can yield
    more relevant variables according t ""
        """
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
            vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        return vips

    vips = vip(model)
    df_importance = pd.DataFrame(list(zip(vips,x_cols)), columns=['vip','col'])
    df_importance['is_important'] = df_importance.vip > 0.9


for n in range(1, len(x_cols)):
    print(n)
    pls_model(n)f


