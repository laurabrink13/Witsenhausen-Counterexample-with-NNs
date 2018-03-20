#python
METARESULTSFILEPATH = "2d/results/meta_basic_cloud1"
meta_results_file = METARESULTSFILEPATH + '.txt'


with open(meta_results_file, 'r') as f:
   data = f.readlines()
   f.close()

for line in data[1:]:

clean_line = line.replace('\n', '')
#     print(clean_line)
k, dat_path, checkpoint_path, avg_cost_test_rep, _ = clean_line.split(' ')

if int(k) in repeat_run_list:
	with open(dat_path, 'rb') as f:
		hyperparameters_dict = pickle.load(f)
		f.close()
	# for key in hyperparameters_dict.keys()
	print(hyperparameters_dict.keys())

