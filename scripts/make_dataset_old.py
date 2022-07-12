
#################
import numpy as np
from pathlib import Path
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import dask_cudf
import argparse
import yaml
from pprint import pprint
# import os
# import time

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='path to .yaml file')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

################ 
if __name__ == '__main__':

    # load yaml file
    opt = parse_opt()
    print(f"loading yaml file...")
    print(opt.cfg)
    config_yaml_data = yaml.safe_load(open(opt.cfg, 'r'))
    pprint(config_yaml_data)

    # set variables from yaml file
    input_data_path = config_yaml_data['input_data_path'] #
    unique_classes_data_path = config_yaml_data['unique_classes_data_path'] #
    number_of_classes = config_yaml_data['number_of_classes'] # the number of classes to randomly select
    size_per_class = config_yaml_data['size_per_class'] # the number of training examples to randomly select per class
    rand_seed = config_yaml_data['rand_seed'] # default is 42
    do_rand_seed = config_yaml_data['do_rand_seed'] # default is true
    tgt_col = config_yaml_data['tgt_col'] # column name of the labels in df
    inp_col = config_yaml_data['inp_col'] # column name of the inputs in df
    do_unknown_class = config_yaml_data['do_unknown_class'] # default is True
    name_for_unknown_class = config_yaml_data['name_for_unknown_class'] # default is _UNKOWN_
    cur_data_split = config_yaml_data['cur_data_split'] # either 'train' 'val' or 'test'
    train_data_split = config_yaml_data['train_data_split'] # proportion of data to use for training
    val_data_split = config_yaml_data['val_data_split'] # proportion of data to use for validation
    test_data_split = config_yaml_data['test_data_split'] # proportion of data to use for testing
    project_name = config_yaml_data['project_name'] # string
    output_data_path = config_yaml_data['output_data_path'] # string
    CUDA_VISIBLE_DEVICES = config_yaml_data['CUDA_VISIBLE_DEVICES']
    do_cuda_vis_dev = config_yaml_data['do_cuda_vis_dev']
    partition_size = config_yaml_data['partition_size']
    
    original_size_per_class = config_yaml_data['size_per_class']

    # if val or test, decrease the train_sizes_per_class proportional to data split size
    if cur_data_split != 'train':
        assert (cur_data_split == 'val') | (cur_data_split == 'test'); "cur_data_split split must be 'train' 'val' or 'test'"
        if cur_data_split != 'val':
            size_per_class /= train_data_split
            size_per_class *= val_data_split
            size_per_class = int(round(size_per_class))
        if cur_data_split != 'test':
            size_per_class /= train_data_split
            size_per_class *= test_data_split
            size_per_class = int(round(size_per_class))

    print(f"train size per class: {size_per_class}")
    # turn off random seed if needed
    if not do_rand_seed:
        rand_seed = None

    print(f"starting Dask GPU cluster...")
    if do_cuda_vis_dev:
        cluster = LocalCUDACluster(
            protocol="ucx",
            enable_tcp_over_ucx=True,
            CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
            )
    else:
        cluster = LocalCUDACluster(
            protocol="ucx",
            enable_tcp_over_ucx=True,
            )
    client = Client(cluster)

    print(f"loading classes file...")
    # make all varaiables needed: number of samples from included df | outpath
    # select classes
    selected_classes = dask_cudf.read_csv(unique_classes_data_path, header=None, names=['classes'])['classes'].compute().to_numpy()
    total_classes = len(selected_classes)
    print(f"randomly selecting {number_of_classes} of {total_classes} classes")
    if do_rand_seed:
        np.random.seed(rand_seed)
    selected_classes = np.random.choice(selected_classes, number_of_classes,replace=False)
    # calc number of samples to take
    # num_incl_samples = selected_classes.shape[0] * size_per_class
    # print(f"number of samples to be taken: {num_incl_samples}")

    print(f"creating folder name...")
    # create output folder 
    if do_unknown_class:
        folder_name = f"{project_name}_{cur_data_split}_classes_{number_of_classes}_and_1_train_size_{original_size_per_class}"
    else:
        folder_name = f"{project_name}_{cur_data_split}_classes_{cur_num_classes}_train_size_{original_size_per_class}"
    out_folder_path = f"{output_data_path}/{folder_name}"
    print(f"creating folder: {out_folder_path}")
    Path(out_folder_path).mkdir( parents=True, exist_ok=True)

    print(f"loading in data")
    # load in df
    df = dask_cudf.read_parquet(input_data_path, partition_size=partition_size)

    # define some functions for mapping
    # def get_incl_clses(df):
    #     bool_mask = df[tgt_col].isin(selected_classes)
    #     return df.loc[bool_mask]
    # def get_excl_clses(df):
    #     bool_mask = df[tgt_col].isin(selected_classes)
    #     return df.loc[~bool_mask]
    def add_unknown_class(df):
        bool_mask = df[tgt_col].isin(selected_classes)
        df.loc[~bool_mask, tgt_col] = name_for_unknown_class
        return df

    if do_unknown_class:
        print(f"adding unknown class name to list")
        selected_classes = np.sort(np.append(name_for_unknown_class, selected_classes))
        df = df.map_partitions(add_unknown_class)
    else:
        print(f"sorting selected class names")
        selected_classes = np.sort(selected_classes)
    
    print(f"performing random selections per class...")
    out_df = []
    selected_classes_str = ''
    total_classes = len(selected_classes)
    for cur_class_i, cur_class in enumerate(selected_classes):
        def get_class(df, cur_class):
            return df[df[tgt_col]==cur_class]
        print(f"\tclass {cur_class_i+1} of {total_classes}: {cur_class}")
        temp_ddf = df.map_partitions(get_class, cur_class).copy() # sub set for our current class
        temp_row_cnt = len(temp_ddf) # get the number of observations
        cur_sample_amt = min([size_per_class, temp_row_cnt])
        print(f"\ttarget sample size is {cur_sample_amt}")
        keep_frac = float(cur_sample_amt / temp_row_cnt)
        temp_ddf = temp_ddf.sample(frac=keep_frac, replace=False, random_state=rand_seed)
        out_df.append(temp_ddf.copy())
        del temp_ddf
        selected_classes_str += cur_class
        selected_classes_str += '\n'
    print(f"concating dataframes...")
    out_df = dask_cudf.concat(out_df, interleave_partitions=True)
    del df
        
    print(f"saving output data...")
    out_df.to_parquet(f"{out_folder_path}/data.parquet")

    print(f"saving new classes file...")
    open(f"{out_folder_path}/class_names.csv", 'w').write(selected_classes_str)

    print(f"saving classes count file...")
    out_df_tgts = out_df[tgt_col].copy()
    del out_df
    out_df_tgts.value_counts().to_frame().to_csv(f"{out_folder_path}/class_counts.csv", header=False, single_file=True)
    del out_df_tgts

    print(f"shutting down Dask client")
    client.shutdown()
    client.close()
    print(f"finished")