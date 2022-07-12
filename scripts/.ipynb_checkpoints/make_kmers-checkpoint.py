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
    input_col_name = config_yaml_data['input_col_name']
    k_mer = config_yaml_data['k_mer']
    possible_gene_values = config_yaml_data['possible_gene_values']  #
    possible_gene_values = sorted(possible_gene_values)
    max_seq_len = config_yaml_data['max_seq_len']
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

    # define functions
    replace_gene_values = []
    for gene_val in possible_gene_values:
        replace_gene_values.append(gene_val + ' ')

    def add_whitespace(df):
        df[input_col_name] = df[input_col_name].str.replace(possible_gene_values, replace_gene_values, regex=False)
        return df

    def get_kmers(df):
        df['temp'] = df[input_col_name].copy()
        df['temp'] = ' '
        for i in np.arange(0, df[input_col_name].str.len().max() - k_mer):
            # print(i)
            temp_df = df[input_col_name].str[i: i + k_mer].fillna(' ')
            change_mask = temp_df.str.len() < k_mer
            temp_df[change_mask] = ' '
            df['temp'] = df['temp'] + ' ' + temp_df
        df['temp'] = df['temp'].str.normalize_spaces()
        df[input_col_name] = df['temp']
        df = df.drop(columns=['temp'])
        return df

    # first we create the Dask dataframe
    print(f"reading file {in_file}")
    df = dask_cudf.read_parquet(in_file,  # location of clean file
                            partition_size=partition_size,
                            )

    # next, we apply the function defined above to the data
    if k_mer == 1:
        df = df.map_partitions(add_whitespace)
    elif (k_mer > 1):
        df = df.map_partitions(get_kmers)

    df[input_col_name] = df[input_col_name].str.split()

    cur_out_path = f"{output_dir}/{project_name}/{out_name}.parquet"
    Path(cur_out_path).mkdir(parents=True, exist_ok=True)
    print(f"saving file {cur_out_path}")
    df.to_parquet(cur_out_path)

    del df

    print(f"shutting down Dask client")
    client.shutdown()
    client.close()
    print(f"finished")