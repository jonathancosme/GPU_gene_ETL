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
    raw_fasta_file = config_yaml_data['raw_fasta_file']  #
    clean_fasta_file = config_yaml_data['clean_fasta_file']  #
    base_col_names = config_yaml_data['base_col_names']  #
    fasta_sep = config_yaml_data['fasta_sep']  #
    CUDA_VISIBLE_DEVICES = config_yaml_data['CUDA_VISIBLE_DEVICES']  #
    do_cuda_vis_dev = config_yaml_data['do_cuda_vis_dev']  #
    partition_size = config_yaml_data['partition_size']  #

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
    print(f"reading .fasta file {raw_fasta_file}")
    df = dask_cudf.read_csv(raw_fasta_file,  # location of raw file
                            sep=fasta_sep,  # this is the '>' sign
                            names=base_col_names,  # column names
                            dtype=str,  # data type
                            partition_size=partition_size,
                            )

    # now we have to shift the data, in order to correct the wrong offset
    print(f"shifting data...")
    df['label'] = df['label'].shift()

    # finally, we drop all empty rows, and reset the index
    print(f"dropping empty rows...")
    df = df.dropna().reset_index(drop=True)

    print(f"saving cleaned data to {clean_fasta_file}")
    Path(clean_fasta_file).mkdir(parents=True, exist_ok=True)
    # the final step is to save the cleaned data.
    _ = df.to_parquet(clean_fasta_file)

    del df

    print(f"shutting down Dask client")
    client.shutdown()
    client.close()
    print(f"finished")