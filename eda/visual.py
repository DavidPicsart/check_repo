import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holoviews as hv
from holoviews import opts, dim


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


def catplot(df, x_col, y_col, hue=None, col=None, row=None, kind="violin", fig_h=10, aspect=0.8, axis_labels_size=10,
            title=None, title_size=18):
    """
    Draws the graph that shows the relationship between mainly categorical variables
    :param df:
    :param x_col: categorical variable
    :param y_col: numeric variable
    :param hue: categorical variable
    :param col: additional categorical variables that will determine the faceting of the grid rowwise
    :param row: additional categorical variables that will determine the faceting of the grid columnwise
    :param fig_h:
    :param aspect:
    :param kind: violin, boxplot, point, bar, strip
    :param axis_labels_size:
    :param title:
    :param title_size:
    :return:
    """
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
    plt.show()


def relplot(df, x_col, y_col, hue=None, col=None, row=None, style=None, size=None, kind="scatter", fig_h=10,
                aspect=0.8, axis_labels_size=10, title=None, title_size=18):
    """
    Draws the graph that shows the relationship between mainly numeric variables
    :param df:
    :param x_col: numeric variable
    :param y_col: numeric variable
    :param hue: Grouping variable that will produce elements with different colors.
    Can be either categorical or numeric, although color mapping will behave differently in latter case.
    :param col: additional categorical variables that will determine the faceting of the grid row wise
    :param row: additional categorical variables that will determine the faceting of the grid column wise
    :param style: Grouping variable that will produce elements with different styles.
    Can have a numeric dtype but will always be treated as categorical.
    :param size: Grouping variable that will produce elements with different styles.
    Can have a numeric dtype but will always be treated as categorical.
    :param kind: scatter, line
    :param fig_h:
    :param aspect:
    :param axis_labels_size:
    :param title:
    :param title_size:
    :return:
    """
    g = sns.relplot(data=df, x=x_col, y=y_col, hue=hue, col=col, row=row, style=style, size=size, kind=kind,
                    height=fig_h, aspect=aspect)
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
    plt.show()


def lmplot(df, x_col, y_col, hue=None, col=None, row=None, fig_h=10,
                aspect=0.8, axis_labels_size=10, title=None, title_size=18):
    """
    Draws scatter plot that fit regression lines to show relationship between noisy variables
    :param df:
    :param x_col: numeric variable
    :param y_col: numeric variable
    :param hue: Grouping variable that will produce elements with different colors.
    Can be either categorical or numeric, although color mapping will behave differently in latter case.
    :param col: additional categorical variables that will determine the faceting of the grid row wise
    :param row: additional categorical variables that will determine the faceting of the grid column wise
    :param fig_h:
    :param aspect:
    :param axis_labels_size:
    :param title:
    :param title_size:
    :return:
    """
    g = sns.lmplot(data=df, x=x_col, y=y_col, hue=hue, col=col, row=row, size=size, height=fig_h, aspect=aspect)

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
    plt.show()


def chordplot(df, source_col, target_col, fig_size=200):
    """
    Draws graph of funnels from one categorical variable (source) to another (target).
    Requires that category names in variables to be different.

    :param df:
    :param source_col:
    :param target_col:
    :param fig_size:
    :return:
    """
    hv.extension('bokeh')
    hv.output(size=fig_size)
    df = df.groupby([source_col, target_col]).size().reset_index(name='value').\
        rename(columns={source_col: "source", target_col: "target"})
    chord = hv.Chord(df)
    chord = chord.opts(
        opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(),
                   labels='index', node_color=dim('index').str()))
    return chord



data = pd.read_csv("/home/david/Downloads/Events_Next_Week.csv")
data['next_week_return'] = data['Event_Count_Next_Week'].apply(lambda x: "active return" if (x >= 20) else ("return" if (x >= 10 and x < 20) else "non return"))
data['active_editor'] = data['editor_done'].apply(lambda x: "active editor" if (x >= 10) else ("editor" if (x >= 5 and x < 10) else "non editor"))
data['active_viewer'] = data['photo_view'].apply(lambda x: 1 if (x >= 4) else 0)
data['magic_lover'] = data['edit_magic_try'].apply(lambda x: 1 if (x >= 6) else 0)




