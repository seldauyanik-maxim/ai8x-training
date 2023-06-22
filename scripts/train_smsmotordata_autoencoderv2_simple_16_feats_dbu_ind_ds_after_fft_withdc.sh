#!/bin/sh
#python train.py --deterministic --regression --print-freq 500 --epochs 50 --optimizer Adam --lr 0.001 --wd 0 --model ai85autoencodersimple --use-bias --dataset SMSMotorData_dur_0_5_overlap_0_75_ForTrain --device MAX78000 --batch-size 64 --validation-split 0 --show-train-accuracy full --qat-policy policies/qat_policy_smsmotordata.yaml "$@"

python train.py --deterministic --regression --print-freq 500 --epochs 100 --optimizer Adam --lr 0.001 --wd 0 --model ai85autoencodersimplev2_16feats --use-bias --dataset SMSMotorData_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_withdc_ForTrain --device MAX78000 --batch-size 64 --validation-split 0 --show-train-accuracy full --qat-policy policies/qat_policy_smsmotordata_longer.yaml "$@"
