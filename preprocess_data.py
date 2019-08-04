import sys
sys.path.append("/jet/prs/workspace/rxrx1-utils")
from rxrx import io as rio

#import settings_model

def rxrx_all():
    print("Loading metadata...")
    # load metadata
    df = rio.combine_metadata().reset_index()
    # create img_id column to match image names
    df["img_id"] = df["dataset"] + "_" + df["id_code"] + "_s" + df["site"].astype("str")
    print("Metadata shape:", df.shape)
        
    # split metadata and shuffle with sample frac=1
    n_debug = 24
    df_train = df[df["dataset"]=="train"].sample(frac=1., random_state=42)
    df_train_debug = df[(df["dataset"]=="train") & (df["sirna"].isin([1110, 1120]))].sample(n=n_debug, random_state=42)
    df_valid = df[(df["dataset"]=="test") & (df["well_type"].isin(["positive_control", "negative_control"]))]\
                .sample(frac=1., random_state=42)
    df_valid_debug = df[(df["dataset"]=="test") & (df["well_type"].isin(["positive_control", "negative_control"])) & 
                        (df["sirna"].isin([1110, 1120])) ].sample(n=n_debug, random_state=42)
    df_test = df[(df["dataset"]=="test") & (df["well_type"]=="treatment")].sample(frac=1., random_state=42)
    # assign -99 class label to test dataset
    df_test["sirna"] = -99
    # create data dictionary
    data = {}
    data["ids_train"] = df_train["img_id"].values.tolist()
    data["labels_train"] = df_train["sirna"].astype("int").astype("str").values
    data["ids_train_debug"] = df_train_debug["img_id"].values.tolist()
    data["labels_train_debug"] = df_train_debug["sirna"].astype("int").astype("str").values
    data["ids_valid"] = df_valid["img_id"].values.tolist()
    data["labels_valid"] = df_valid["sirna"].astype("int").astype("str").values
    data["ids_valid_debug"] = df_valid_debug["img_id"].values.tolist()
    data["labels_valid_debug"] = df_valid_debug["sirna"].astype("int").astype("str").values
    data["ids_test"] = df_test["img_id"].values.tolist()
    data["labels_test"] = df_test["sirna"].astype("int").astype("str").values
    
    # print numbers check
    check_total = 0
    for key in data.keys():
        check_total += len(data[key])
        print("{} values in {}".format(len(data[key]), key))
    print("Total ids: {}".format(check_total/2))
    
    # calculate normalization factors by experiment
    print("Calculating normalization factors by experiment...")
    df_stats = rio._load_stats()
    df_norm = df_stats.groupby(["experiment", "channel"])[["median", "std"]].mean().unstack().reset_index()
    exp_norm_dict = {}
    for exp in df_norm["experiment"].values:
        exp_norm_dict[exp] = {"median" : df_norm.loc[df_norm["experiment"]==exp, "median"].values, 
                              "std" : df_norm.loc[df_norm["experiment"]==exp, "std"].values}
    data["exp_norm_dict"] = exp_norm_dict    
        
    return data

def rxrx_control():
    return None

def rxrx_treatment():
    return None


    