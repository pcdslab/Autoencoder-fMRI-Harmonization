#!/bin/bash
#SBATCH --job-name=combat_all
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --nodelist=dragon-compute-04
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
# Outputs ----------------------------------
#SBATCH --output=batch_logs/%x-%j.out
#SBATCH --error=batch_logs/%x-%j.err
# ------------------------------------------
pwd; hostname; date
cd ..
cd code/

# proportional split
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE-10-fold --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE-10-fold --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE-10-fold --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE-10-fold --ae_type DAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI

#########################################

### Center classification using all ML methods, RF, NB and SVM
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Trinity,SBL
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=KKI,OHSU
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Leuven,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UM,USM
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Olin,UCLA
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Caltech,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Pitt,Yale

# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Trinity,SBL
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=KKI,OHSU
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Leuven,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UM,USM
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Olin,UCLA
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Caltech,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Pitt,Yale

# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Trinity,SBL
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=KKI,OHSU
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Leuven,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UM,USM
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Olin,UCLA
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Caltech,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Pitt,Yale

#########################################

# center classification using NT subjects and AE harmonization
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI

# center classification using NT subjects with and without combat 
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Trinity,SBL
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=KKI,OHSU
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Leuven,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UM,USM
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Olin,UCLA
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Caltech,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Pitt,Yale

#########################################

# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Trinity,SBL
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=KKI,OHSU
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Leuven,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UM,USM
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Olin,UCLA
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Caltech,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method NB --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Pitt,Yale

# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Trinity,SBL
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=KKI,OHSU
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Leuven,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UM,USM
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Olin,UCLA
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Caltech,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ae_type AE --ml_method SVM --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Pitt,Yale

#########################################

### Run center classification with harmonization using AEs, change --ae_type tp TAE and SAE
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Trinity,SBL
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=KKI,OHSU
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Leuven,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UM,USM
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Olin,UCLA
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Caltech,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Pitt,Yale

# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,KKI
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UCLA,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Trinity,SBL
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=KKI,OHSU
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Leuven,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=UM,USM
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Olin,UCLA
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Caltech,MaxMun
# python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-ae-harmonization --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 5 --centers=Pitt,Yale

#########################################

# leave one site out experiments
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale

# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type TAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale

# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
# python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale

#########################################


# leave one site out using combat and without combat, just change run_combat = False or True in line 121
# python asd_harmonization_AEs.py --p_method ASD-ml --ae_type SAE --ml_method RF --path /lclhome/falmu027/projects/workshop_paper/samples/10_CV_1035.csv --output output_mix_match/ --fold 10 --centers=Pitt,Yale
