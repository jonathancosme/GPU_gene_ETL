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
    project_name = config_yaml_data['project_name']  #
    out_name = config_yaml_data['out_name'] 
    rand_seed = config_yaml_data['rand_seed']  #
    do_rand_seed = config_yaml_data['do_rand_seed']  #
    train_split = config_yaml_data['train_split']  #
    val_split = config_yaml_data['val_split']  #
    test_split = config_yaml_data['test_split']  #
    CUDA_VISIBLE_DEVICES = config_yaml_data['CUDA_VISIBLE_DEVICES']  #
    do_cuda_vis_dev = config_yaml_data['do_cuda_vis_dev']  #
    partition_size = config_yaml_data['partition_size']  #

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

    # first we create the Dask dataframe
    print(f"reading file {in_file}")
    df = dask_cudf.read_parquet(in_file,  # location of clean file
                            partition_size=partition_size,
                            )

    print(f"splitting data...")
    train_df, val_df, test_df = df.random_split([train_split, val_split, test_split], random_state=rand_seed)

    proj_dir = f"{output_dir}/{project_name}"
    Path(proj_dir).mkdir(parents=True, exist_ok=True)
    
    output_name = f"{proj_dir}/{out_name}_train.parquet"
    print(f"saving train file {output_name}")
    _ = train_df.to_parquet(output_name)
    
    output_name = f"{proj_dir}/{out_name}_val.parquet"
    print(f"saving val file {output_name}")
    _ = val_df.to_parquet(output_name)
    
    output_name = f"{proj_dir}/{out_name}_test.parquet"
    print(f"saving val file {output_name}")
    _ = test_df.to_parquet(output_name)

    del df, train_df, val_df, test_df

    print(f"shutting down Dask client")
    client.shutdown()
    client.close()
    print(f"finished")