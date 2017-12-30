import matplotlib.pyplot as plt
import numpy

def history(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def curves(y, yhat, labels):
    plt.plot(y, label=labels[0])
    plt.plot(yhat, label=labels[1])
    plt.legend()
    plt.show()


def setup_plot(title):
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


def prediction(Y, Yhat, title):
    setup_plot(title)
    # place the prediction as we did with test_values
    for idx in range(len(Yhat)):
        #idx += 1
        x = idx
        y = Yhat[idx]
        if idx > 0:
            yhat_trend = numpy.sign(Yhat[idx]-Y[idx-1])
            y_trend = numpy.sign(Y[idx]-Y[idx-1])
            error = int(yhat_trend != y_trend)
            color = 'red' if error is 1 else 'green'
        else:
            color = 'green'
        plt.plot([x], [y], marker='o', markersize=5, color=color)
    plt.plot(Yhat, marker='', color='r', linewidth=1)
    plt.plot(Y, marker='o', color='b', linewidth=1.0, alpha=0.5)
    plt.show()


def features(raw_dataset):
    values = raw_dataset.values
    # specify columns to plot
    num_features = raw_dataset.shape[1]
    groups = range(num_features - (num_features % 5))
    i = 1
    # plot each column
    plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
    num_rows = int(len(groups)/2)
    num_cols = int(len(groups)/5)
    for group in groups:
        plt.subplot(num_rows, num_cols, i)
        plt.plot(values[:, group], linewidth=0.2)
        plt.title(raw_dataset.columns[group], y=0.75, loc='left', fontsize=7)
        i += 1
    plt.tight_layout()
    plt.show()
