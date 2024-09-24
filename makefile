copy_data:
	cp ./code_paper_EPSL/additional_data/all_viscosity.hdf5 ./src/gpvisc/data/all_viscosity.hdf5

copy_models:
	cp -r ./code_paper_EPSL/models/GP/GP_model76/* ./src/gpvisc/models/GP_model1/
	cp -r ./code_paper_EPSL/models/GP/GP_model49/* ./src/gpvisc/models/GP_model2/
	cp -r ./code_paper_EPSL/models/GP/GP_model94/* ./src/gpvisc/models/GP_model3/

copy_figures:
	cp ./code_paper_EPSL/figures/Figure1.pdf ../manuscrits/i-Visc/Figure1.pdf
	cp ./code_paper_EPSL/figures/Figure2.pdf ../manuscrits/i-Visc/Figure2.pdf
	cp ./code_paper_EPSL/figures/Figure3.pdf ../manuscrits/i-Visc/Figure3.pdf
	cp ./code_paper_EPSL/figures/extrapolation_paper.pdf ../manuscrits/i-Visc/Figure4.pdf
	cp ./code_paper_EPSL/figures/Planet_temperature_profile.pdf ../manuscrits/i-Visc/Figure5.pdf
	cp ./code_paper_EPSL/figures/outgassing_efficiency.pdf ../manuscrits/i-Visc/Figure6.pdf
	cp ./code_paper_EPSL/figures/Phases_profile_papier.pdf ../manuscrits/i-Visc/Figure7.pdf
	cp ./code_paper_EPSL/figures/viscosity_profile_all.pdf ../manuscrits/i-Visc/Figure8.pdf
	cp ./code_paper_EPSL/figures/Surface_Interior_Horizontal.pdf ../manuscrits/i-Visc/Figure9a.pdf
	cp ./code_paper_EPSL/figures/Surface_Interior_Hard.pdf ../manuscrits/i-Visc/Figure9b.pdf

install:
	python3 -m pip install .
	
train_gp:
	python3 ./code_paper_EPSL/5_gp_train.py --gp_save_name ./code_paper_EPSL/models/test --training_iter 60000 --early_criterion 5000 --training_iter_gp 1000