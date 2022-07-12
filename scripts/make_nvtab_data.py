#################
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import nvtabular as nvt
from glob import glob
import argparse
import yaml
from pprint import pprint
import numpy as np
from pathlib import Path


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
    input_dir = config_yaml_data['input_dir']  #
    input_col_name = config_yaml_data['input_col_name']  #
    label_col_name = config_yaml_data['label_col_name']  #
    max_seq_len = config_yaml_data['max_seq_len']  #
    row_group_size = config_yaml_data['row_group_size']
    CUDA_VISIBLE_DEVICES = config_yaml_data['CUDA_VISIBLE_DEVICES']
    do_cuda_vis_dev = config_yaml_data['do_cuda_vis_dev']

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

    print(f"creating pipeline...")
    # create the pipeline
    # nvt.ColumnGroup(
    cat_features = [input_col_name] >> nvt.ops.Categorify() >> nvt.ops.ListSlice(0, end=max_seq_len, pad=True,
                                                                                 pad_value=0.0)
    lab_features = [label_col_name] >> nvt.ops.Categorify()
    # add label column
    output = cat_features + lab_features
    # create workflow
    workflow = nvt.Workflow(output, client=client)

    workflow_name = f"{input_dir}/workflow"
    data_paths = [x for x in glob(f"{input_dir}/*") if '.parquet' in x]
    train_path = [x for x in data_paths if 'train_data' in x][0]
    val_path = [x for x in data_paths if 'val_data' in x][0]
    test_path = [x for x in data_paths if 'test_data' in x][0]

    # fitting on training data, and saving the workflow
    print("fitting nvtab workflow on training data...")
    workflow.fit(nvt.Dataset(train_path, engine='parquet', row_group_size=row_group_size))
    workflow.save(workflow_name)

    shuffle = nvt.io.Shuffle.PER_PARTITION
    print("making nvtab dataset for training...")
    workflow.transform(nvt.Dataset(train_path, engine='parquet', row_group_size=row_group_size)).to_parquet(
        output_path=f"{input_dir}/train_nvtab.parquet",
        shuffle=shuffle,
        cats=[input_col_name],
        labels=[label_col_name],
    )

    print(f"making nvtab dataset for val...")
    workflow.transform(nvt.Dataset(val_path, engine='parquet', row_group_size=row_group_size)).to_parquet(
        output_path=f"{input_dir}/val_nvtab.parquet",
        shuffle=None,
        out_files_per_proc=None,
        cats=[input_col_name],
        labels=[label_col_name],
    )

    print(f"making nvtab dataset for test...")
    workflow.transform(nvt.Dataset(test_path, engine='parquet', row_group_size=row_group_size)).to_parquet(
        output_path=f"{input_dir}/test_nvtab.parquet",
        shuffle=None,
        out_files_per_proc=None,
        cats=[input_col_name],
        labels=[label_col_name],
    )

    client.cancel(workflow)

    print(f"shutting down Dask client")
    client.shutdown()
    client.close()
    print(f"finished")