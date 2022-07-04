import matplotlib.pyplot as plt
import os
import seaborn as sns


def plot_metric_graph(history, metric =''):
    # summarize history for loss
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+ metric])
    plt.title('model '+metric)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

def plot_label_count(data, target_col = ''):
    sns.countplot(x=target_col, data = data, palette = 'Set3')
    plt.xticks()
    plt.ylabel("Count")
    plt.xlabel("Target")
    plt.title("Distribution of Target")
    plt.show()