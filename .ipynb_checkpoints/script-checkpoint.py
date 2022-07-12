df # a dask cudf dataframe
unique_classes # a list of unique class labels
numbers_of_classes # a list of numbers of classes
train_sizes_per_class # a list of training sizes per class

rand_seed # default is 42
do_rand_seed # default is true

tgt_col # column name of the labels in df
inp_col # column name of the inputs in df

do_unknown_class # default is True
name_for_unknown_class # default is _UNKOWN_

cur_data_split # either 'train' 'val' or 'test'
train_data_split # proportion of data to use for training
val_data_split # proportion of data to use for validation
test_data_split # proportion of data to use for testing

project_name # string

data_output_path # string

######

# sort from largest to smallest
numbers_of_classes = sorted(numbers_of_classes, ascending=False)
train_sizes_per_class = sorted(train_sizes_per_class, ascending=False)

# turn some variables into numpy arrays
numbers_of_classes = np.array(numbers_of_classes)
train_sizes_per_class = np.array(train_sizes_per_class)
unique_classes = np.array(unique_classes)


# if val or test, decrease the train_sizes_per_class proportional to data split size
if cur_data_split != 'train':
	assert (cur_data_split == 'val') | (cur_data_split == 'test'); "cur_data_split split must be 'train' 'val' or 'test'"
	if cur_data_split != 'val':
		train_sizes_per_class /= train_data_split
		train_sizes_per_class *= val_data_split
		train_sizes_per_class = train_sizes_per_class.round().astype(int)
	if cur_data_split != 'test':
		train_sizes_per_class /= train_data_split
		train_sizes_per_class *= test_data_split
		train_sizes_per_class = train_sizes_per_class.round().astype(int)

# turn off random seed if needed
if do_rand_seed:
	rand_seed = None

# create dask cudf of df indexed by labels
df_class_indexed = df.set_index(tgt_col)

for num_classes_i, cur_num_classes in enumerate(numbers_of_classes):
	print("\n")
	print(f"num_classes_i: {num_classes_i}, cur_num_classes: {cur_num_classes}")
	cur_unq_classes = np.random.choice(unique_classes, size=cur_num_classes, replace=False, seed=rand_seed)
	cur_classes_df = df_class_indexed.loc[cur_unq_classes]
	# len_cur_classes_df = cur_classes_df.shape[0]
	# print(f"len_cur_classes_df: {len_cur_classes_df}")
	
	# len_cur_not_class_df = cur_not_class_df.shape[0]
	# print(f"len_cur_not_class_df: {len_cur_not_class_df}")
	for cls_sz_i, cur_cls_sz in enumerate(train_sizes_per_class):
		cur_out_df = []
		print(f"\tselecting from classes...")
		for cur_class_i, cur_class in enumerate(cur_unq_classes):
			print(f"\tcur_class_i: {cur_class_i}, cur_class: {cur_class}")
			cur_class_df = cur_classes_df.loc[cur_class].reset_index()
			cur_class_df = cur_class_df.sample(cur_cls_sz, replace=False, random_state=rand_seed)
			cur_out_df.append(cur_class_df.copy())
		del cur_class_df

		if do_unknown_class:
			print(f"\tselecting from not classes...")
			cur_not_class_df = df_class_indexed.loc[~cur_unq_classes].reset_index().sample(cur_cls_sz, replace=False, random_state=rand_seed)
			cur_not_class_df[tgt_col] = name_for_unknown_class # set the labels of unkown class
			cur_out_df.append(cur_not_class_df.copy())
			del cur_not_class_df

			folder_name = f"{project_name}_{cur_data_split}_classes_{cur_num_classes}_and_1_size_{cur_cls_sz}"
		
		else:
			folder_name = f"{project_name}_{cur_data_split}_classes_{cur_num_classes}_size_{cur_cls_sz}"

		

		cur_out_df = dask_cudf.concat(cur_out_df, ignore_index=True).reset_index(True)
		cur_out_df = cur_out_df.reset_index(shuffle='disk')

		cur_sz_unq_cls = cur_out_df[tgt_col].unique().sorted(ascending=True) # get list of unique class names
		cur_sz_unq_cls_val_cnts = cur_out_df[tgt_col].value_counts() # get a count for each class


		cur_out_path = f"{data_output_path}/{folder_name}"
		print(f"making path {cur_out_path}")
		Path(cur_out_path).mkdir( parents=True, exist_ok=True )

		cur_sz_unq_cls.to_csv(f"{cur_out_path}/class_names.csv", index=False, single_file=True)
		cur_sz_unq_cls_val_cnts.to_csv(f"{cur_out_path}/class_counts.csv", single_file=True)

		cur_out_df.to_parquet(f"{cur_out_path}/data.parquet")


		# create folder for output. name should have data split, number of class (and uknown if applicable) and data size
		# save metadata in folder, including the unique class labels (be sure to add class 0 label for unkown)
		# save parquet data in folder