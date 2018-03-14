declare lr_arr=(0.001, 0.005, 0.01)

## now loop through the above array
for lr in 0.001 0.005 0.01
do
	# echo $lr
   python nn_run.py $lr 20 20 0.5 100 >> test_output.txt
   # or do whatever with individual element of the array
done
