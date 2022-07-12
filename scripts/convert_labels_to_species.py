#################
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import dask_cudf
import argparse
import yaml
from pprint import pprint
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
    in_file = config_yaml_data['in_file']  #
    output_dir = config_yaml_data['output_dir']  #
    project_name = config_yaml_data['project_name']
    out_name = config_yaml_data['out_name']
    label_col_name = config_yaml_data['label_col_name']
    label_regex = config_yaml_data['label_regex']  #
    CUDA_VISIBLE_DEVICES = config_yaml_data['CUDA_VISIBLE_DEVICES']  #
    do_cuda_vis_dev = config_yaml_data['do_cuda_vis_dev']  #
    partition_size = config_yaml_data['partition_size']



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

    # first we create the Dask dataframe
    print(f"reading file {in_file}")
    df = dask_cudf.read_parquet(in_file,  # location of clean file
                            partition_size=partition_size,
                            )

    def extract_labels(df):
        df[label_col_name] = df[label_col_name].str.extract(label_regex).loc[:, 0]
        return df

    df = df.map_partitions(extract_labels)

    cur_out_path = f"{output_dir}/{project_name}"
    Path(cur_out_path).mkdir(parents=True, exist_ok=True)
    cur_out_path = f"{cur_out_path}/{out_name}.parquet"
    print(f"saving file {cur_out_path}")
    df.to_parquet(cur_out_path)

    unq_labs_df = df.sort_values(label_col_name)[label_col_name].unique().to_frame()
    cur_out_path = f"{output_dir}/{project_name}"
    Path(cur_out_path).mkdir(parents=True, exist_ok=True)
    cur_out_path = f"{cur_out_path}/{out_name}_unq_labs.csv"
    print(f"saving file {cur_out_path}")
    _ = unq_labs_df.to_csv(cur_out_path, index=False, header=False, single_file=True)

    del df, unq_labs_df

    print(f"shutting down Dask client")
    client.shutdown()
    client.close()
    print(f"finished")