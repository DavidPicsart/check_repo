import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def pieplot(df, col, fig_h=12, fig_w=12, labels_size=14, legend_size=14, title=None, title_size=22):
    """
    Makes plot of dataframe of "pie" kind
    Args:
        df: dataframe
        col: necessary column to plot pie plot
        fig_h: height of figure
        fig_w: width of figure
        labels_size
        legend_size
        title: title of pie plot
        title_size
    """

    fig, ax = plt.subplots(1, 1, figsize=(fig_h, fig_w))
    labels = df[col].value_counts().index[::]
    ax.pie(data[col].value_counts(), autopct='%1.1f%%', textprops={'fontsize': labels_size})
    ax.set_title(title, fontsize=title_size)
    ax.legend(labels, loc="best", fontsize=legend_size)
    plt.show()


def distplot(df, columns_list, fig_h=12, fig_w=12, legend_size=14, title=None, title_size=22):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plt.title(title, fontsize=title_size)
    for i in columns_list:
        sns.distplot(df[i].values, kde=True, label=i, ax=ax)
    plt.legend(fontsize=legend_size, loc="best")
    plt.show()

def jointplot(df, x_col, y_col, fig_h=10, axis_labels_size=14, title=None, title_size=18):
    g = sns.jointplot(x=x_col, y=y_col, data=df, kind='kde', height=fig_h)
    # Set title
    g.fig.suptitle(title, fontsize=title_size)
    # Set axis labels
    g.set_axis_labels(x_col, y_col, fontsize=axis_labels_size)
    # Format nicely
    g.fig.tight_layout()
    # Reduce plot to make room for title
    g.fig.subplots_adjust(top=0.9)


def catplot(df, x_col, y_col, hue=None, col=None, row=None, fig_h=10, aspect=0.8, kind="violin", axis_labels_size=10,
            title=None, title_size=18):
    g = sns.catplot(x=x_col, y=y_col, hue=hue, col=col, row=row, data=df, kind=kind, height=fig_h, aspect=aspect)
    # Set axis labels
    axes = g.axes.flatten()
    for i in axes:
        i.set_xlabel(x_col, fontsize=axis_labels_size)
        i.set_ylabel(y_col, fontsize=axis_labels_size)
    # Set title
    g.fig.suptitle(title, fontsize=title_size)
    # Format nicely.
    g.fig.tight_layout()
    # Reduce plot to make room for title
    g.fig.subplots_adjust(top=0.9)


data = pd.read_csv("/Users/dk/Documents/Events_Next_Week.csv")
data['next_week_return'] =data['Event_Count_Next_Week'].apply(lambda x: 1 if (x >= 10) else 0)
data['active_editor'] = data['editor_done'].apply(lambda x: 1 if (x >= 5) else 0)
data['active_viewer'] = data['photo_view'].apply(lambda x: 1 if (x >= 4) else 0)
data['magic_lover'] = data['edit_magic_try'].apply(lambda x: 1 if (x >= 6) else 0)

catplot(df=data, x_col="next_week_return", y_col="event_count_first_24h",
                    kind="violin", fig_h=10, aspect=1, title="Graph")
