import pandas as pd


def upsampling(df, labels=None, label_name="label"):
    if not labels:
        labels = list(df[label_name].unique())
    max_length = max(list(df[df[label_name].isin(labels)][label_name].value_counts()))
    ret_df = pd.DataFrame()
    for label in labels:
            tmp_df = df[df[label_name] == label]
            if max_length <= len(tmp_df):
                ret_df = pd.concat([ret_df, tmp_df.sample(n=max_length, replace=False)])
            else:
                ret_df = pd.concat([ret_df, tmp_df.sample(n=len(tmp_df), replace=False)])
                ret_df = pd.concat([ret_df, tmp_df.sample(n=max_length-len(tmp_df), replace=True)])
    return ret_df


def downsampling(df, labels=None, label_name="label"):
    if not labels:
        labels = list(df[label_name].unique())
    min_length = min(list(df[df[label_name].isin(labels)][label_name].value_counts()))
    ret_df = pd.DataFrame()
    for label in labels:
        ret_df = pd.concat([ret_df, df[df[label_name] == label].sample(n=min_length, replace=False)])
    return ret_df


def multi_upsampling(df, label_cluster, label_name="label"):
    max_length = 0
    max_length_index = None
    for i, labels in enumerate(label_cluster):
        for label in labels:
            length = len(df[df[label_name] == label])
            if length > max_length:
                max_length = length
                max_length_index = i
    cluster_length = [max_length * len(_) for _ in label_cluster][max_length_index]
    ret_df = pd.DataFrame()
    for labels in label_cluster:
        length = int(cluster_length / len(labels))
        for label in labels:
            tmp_df = df[df[label_name] == label]
            if length <= len(tmp_df):
                ret_df = pd.concat([ret_df, tmp_df.sample(n=length, replace=False)])
            else:
                ret_df = pd.concat([ret_df, tmp_df.sample(n=len(tmp_df), replace=False)])
                ret_df = pd.concat([ret_df, tmp_df.sample(n=length-len(tmp_df), replace=True)])
    return ret_df


def multi_downsampling(df, label_cluster, label_name="label"):
    min_length = len(df)
    min_length_index = None
    for i, labels in enumerate(label_cluster):
        for label in labels:
            length = len(df[df[label_name] == label])
            if length < min_length:
                min_length = length
                min_length_index = i
    cluster_length = max([min_length * len(_) for _ in label_cluster])
    ret_df = pd.DataFrame()
    for labels in label_cluster:
        length = int(cluster_length / len(labels))
        for label in labels:
            if length == len(df[df[label_name] == label]):
                ret_df = pd.concat([ret_df, df[df[label_name] == label].sample(n=length, replace=False)])
            else:
                ret_df = pd.concat([ret_df, df[df[label_name] == label].sample(n=length, replace=False)])
    return ret_df



df = pd.DataFrame({"field1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "field2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "label": ["a", "a", "a", "b", "c", "c", "c", "c", "c", "c"]})
print("upsampling\n", upsampling(df))
print("downsampling\n", downsampling(df))
print("multi_upsampling A\n", multi_upsampling(df, [["a"], ["b", "c"]]))
print("multi_upsampling B\n", multi_upsampling(df, [["a", "b"], ["c"]]))
print("multi_downsampling A\n", multi_downsampling(df, [["a"], ["b", "c"]]))
print("multi_downsampling B\n", multi_downsampling(df, [["a", "b"], ["c"]]))
