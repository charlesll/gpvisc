train_one:
	python3 ./src/Training_single.py

ray:
	python3 ./src/ray_opt.py > ./model/ray3.log

select:
	python3 ./src/ray_select.py

transfer:
	tar czvf compressed_files.tar.gz ./data/* ./src/* ./makefile
	scp compressed_files.tar.gz lelosq@stellagpu01:/gpfs/users/lelosq/ivisc/

transfer_back:
	scp lelosq@stellagpu01:/gpfs/users/lelosq/ivisc/models/l4_n400_p0.15_GELU_cpfree.pth ~/ownCloud/IVIMAP_shared/ivisc/models/

decompress:
	tar xzvf compressed_files.tar.gz
	