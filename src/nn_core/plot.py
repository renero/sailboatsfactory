import matplotlib.pyplot as plt
import numpy


import data


def history(history):
    plt.plot(history.history['loss'], label='train')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def curves(y, yhat, labels):
    plt.plot(y, label=labels[0])
    plt.plot(yhat, label=labels[1])
    plt.legend()
    plt.show()


def setup_plot(title):
    #plt.style.use('fivethirtyeight')
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    major_ticks = numpy.arange(0, 1001, 10)
    minor_ticks = numpy.arange(0, 1001, 2)
    ax = plt.gca()  # Get Current Axes
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.xaxis.grid(True, which='major', color='grey', linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, which='minor', color='grey', linestyle='--', alpha=0.2)
    ax.yaxis.grid(True, which='major', color='grey', linestyle='--', alpha=0.5)
    ax.yaxis.grid(True, which='minor', color='grey', linestyle='--', alpha=0.2)
    plt.minorticks_on()
    plt.ylabel('price')
    plt.title(title)
    return ax


def prediction(Y, Yhat, rmse, n_errs, params):
    title = 'RMSE={:.02f}, T.E={:.02f}% ({:d}/{:d})'.format(
        rmse, (n_errs/(len(Yhat)-1)), n_errs, len(Yhat) - 1)
    ax = setup_plot(title)
    # place the prediction as we did with test_values
    for idx in range(len(Yhat)):
        x = idx
        y = Yhat[idx]
        if idx > 0:
            yhat_trend = numpy.sign(Yhat[idx]-Y[idx-1])
            y_trend = numpy.sign(Y[idx]-Y[idx-1])
            error = int(yhat_trend != y_trend)
            color = 'red' if error is 1 else 'green'
            plt.plot([x], [y], marker='+', markersize=8, color=color)
    #    else:
    #        color = 'green'
    plt.plot(Yhat, '--', marker='', color='r', linewidth=0.4, alpha=0.8)
    ax.plot(Y, color='b', lw=1, alpha=0.7)
    ax.plot(Y, color='b', marker='.', alpha=0.4, markersize=8)
    plt.show()


def original(yraw, yhat, params):
    """
    Plots the original values against the predicted ones.
    To restore original values, 1st unscale, then undiff, and finally
    unlog (exp1m), which is the sequence of steps in reverse order.
    """
    ax = setup_plot('Original price and prediction levels')
    yhat = params['y_scaler'].inverse_transform(yhat.reshape(-1, 1))
    yhat = data.inverse_diff(yhat.reshape(-1, 1), yraw)
    yhat = numpy.expm1(yhat)
    ax.plot(numpy.expm1(yraw), color='b', lw=1, alpha=0.7)
    ax.plot(numpy.expm1(yraw), color='b', marker='.', markersize=8, alpha=0.3)
    plt.plot(yhat, '_', color='r', alpha=0.6)
    plt.show()


def features(raw_dataset):
    values = raw_dataset.values
    # specify columns to plot
    num_features = raw_dataset.shape[1]
    groups = range(num_features - (num_features % 5))
    i = 1
    # plot each column
    plt.figure(num=None, figsize=(12, 12), dpi=80,
               facecolor='w', edgecolor='k')
    num_rows = int(len(groups)/2)
    num_cols = int(len(groups)/5)
    for group in groups:
        plt.subplot(num_rows, num_cols, i)
        plt.plot(values[:, group], linewidth=0.2)
        plt.title(raw_dataset.columns[group], y=0.75, loc='left', fontsize=7)
        i += 1
    plt.tight_layout()
    plt.show()
