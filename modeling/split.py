import pandas as pd
from pyspark.sql import functions as F
from pyspark import SQLContext
from pyspark import SparkContext


# TODO add timestamp splitting strategy

class PercentageStrategyEntity:
    """
    Splits in chunks by entities of data frame.
    Returns dict where keys are the names of splits (e.g. train etc) and values are respective data frames.
    """
    @staticmethod
    def split(data, split_entity, train_pct, val_pct, test_pct=0, seed=0, lib="pandas"):
        assert round(train_pct + val_pct + test_pct, 2) == 1.0, "Split percentages should add up to 1"
        assert lib == "pandas" or lib == "pyspark", "Not a correct name of lib. Should be either pandas or pysprak."

        # update pcts for every chunk to account for reduced size data chunks because of previous splits
        pcts = [train_pct, round(val_pct/(1+1.e-5-train_pct), 2), round(test_pct/(1+1.e-5-train_pct-val_pct), 2)]
        chunks_names = ["train", "val", "test"]

        if lib == "pandas":
            unique_entities = pd.DataFrame(data[split_entity].unique(), columns=[split_entity])
            splits_dict = {}
            for ind, i in enumerate(pcts):
                if i > 0:
                    # sample from df of unique entities, join to data and append it to dict
                    unique_entities_sample = unique_entities.sample(frac=i, replace=False, random_state=seed)
                    sample_data = data.merge(unique_entities_sample, on=split_entity, how="inner")
                    splits_dict.update({chunks_names[ind]: sample_data})
                    # filter out already used entities from df of unique entities
                    unique_entities = pd.DataFrame(set(unique_entities[split_entity]) -
                                                set(unique_entities_sample[split_entity]), columns=[split_entity])

        elif lib == "pyspark":
            data.persist()
            # create df of unique entities and empty dict for data chunks
            unique_entities = data.select(split_entity).distinct()
            splits_dict = {}
            for ind, i in enumerate(pcts):
                if i > 0:
                    # sample from df of unique entities, join to data and append it to dict
                    unique_entities_sample = unique_entities.sample(withReplacement=False, fraction=i, seed=seed)
                    sample_data = data.join(unique_entities_sample, on=split_entity, how="inner")
                    splits_dict.update({chunks_names[ind]: sample_data})
                    # filter out already used entities from df of unique entities
                    unique_entities = unique_entities.join(unique_entities_sample, on=split_entity, how="leftanti")

        return splits_dict


class PercentageStrategyRow:
    """
    Splits in chunks by rows of data frame.
    Returns dict where keys are the names of splits (e.g. train etc) and values are respective data frames.
    """
    @staticmethod
    def split(data, train_pct, val_pct, test_pct=0, seed=0, lib="pandas"):
        assert round(train_pct + val_pct + test_pct, 2) == 1.0, "Split percentages should add up to 1"
        assert lib == "pandas" or lib == "pyspark", "Not a correct name of lib. Should be either pandas or pysprak."

        # update pcts for every chunk to account for reduced size data chunks because of previous splits
        pcts = [train_pct, round(val_pct/(1+1.e-5-train_pct), 2), round(test_pct/(1+1.e-5-train_pct-val_pct), 2)]
        chunks_names = ["train", "val", "test"]

        if lib == "pandas":
            splits_dict = {}
            for ind, i in enumerate(pcts):
                if i > 0:
                    # sample from data and append sample to dict
                    sample_data = data.sample(frac=i, replace=False, random_state=seed)
                    splits_dict.update({chunks_names[ind]: sample_data})
                    # filter out sample rows from data
                    data = pd.concat([data, sample_data]).drop_duplicates(keep=False)

        elif lib == "pyspark":
            data.persist()
            # create row number for consequent filtrations
            data = data.withColumn("row", F.monotonically_increasing_id())
            # create df of unique entities and empty dict for data chunks
            splits_dict = {}
            for ind, i in enumerate(pcts):
                if i > 0:
                    # sample from df of unique entities, join to data and append it to dict
                    sample_data = data.sample(withReplacement=False, fraction=i, seed=seed)
                    splits_dict.update({chunks_names[ind]: sample_data.drop("row")})
                    # filter out already used entities from df of unique entities
                    data = data.join(sample_data, on="row", how="leftanti")

        return splits_dict


















class ProportionStrategy:

    @staticmethod
    def split(data):
        pass

# start spark context
sc = SparkContext(appName="PythonCollaborativeFilteringExample")
sqlCtx = SQLContext(sc)
source = sqlCtx.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
    '/home/david/Documents/Recommendations_DL/search_retrieval/data/activity_data_sample_1000_check.csv')
unique_items = source.select("search_sid").distinct()
unique_items_sample = unique_items.sample(False, 0.3, seed=0)
unique_items_leftover = unique_items.join(unique_items_sample, on = "item_id", how="leftanti")


data = pd.read_csv('/home/david/Documents/Recommendations_DL/search_retrieval/data/activity_data_sample_1000_check.csv')