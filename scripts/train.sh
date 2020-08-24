#!/bin/bash

if [ $# -ne 3 -a $# -ne 4 ]; then
    echo "Usage: $0 DATASET MODEL OUT [OPTION]" 1>&2
    exit 1
fi

DATASET=$1
MODEL=$2
OUT=$3
if [ $# -eq 4 ]; then
    OPTION=$4
else
    OPTION=""
fi

# Set up dataset
case ${DATASET} in
    "cifar10" )
	OPTION="${OPTION} --dataset CIFAR10" ;;
    "cifar10ag25" )
	OPTION="${OPTION} --dataset CIFAR10AG --noise_scale 25" ;;
    "cifar10ag5-50" )
	OPTION="${OPTION} --dataset CIFAR10AG --noise_scale 5 --noise_scale_high 50" ;;
    "cifar10lg25p16" )
	OPTION="${OPTION} --dataset CIFAR10LG --noise_scale 25 --patch_size 16" ;;
    "cifar10lg25p8-24" )
	OPTION="${OPTION} --dataset CIFAR10LG --noise_scale 25 --patch_size 8 --patch_max_size 24" ;;
    "cifar10u50" )
	OPTION="${OPTION} --dataset CIFAR10U --noise_scale 50" ;;
    "cifar10mix" )
	OPTION="${OPTION} --dataset CIFAR10MIX" ;;
    "cifar10bg25k5" )
	OPTION="${OPTION} --dataset CIFAR10BG --noise_scale 25 --kernel_size 5" ;;
    "cifar10abg25k5" )
	OPTION="${OPTION} --dataset CIFAR10ABG --noise_scale 25 --kernel_size 5" ;;
    "cifar10mg25" )
	OPTION="${OPTION} --dataset CIFAR10MG --multi_noise_scale 25"
	PRIOR=multiplicative_gaussian ;;
    "cifar10mg5-50" )
	OPTION="${OPTION} --dataset CIFAR10MG --multi_noise_scale 5 --multi_noise_scale_high 50"
	PRIOR=multiplicative_gaussian ;;
    "cifar10amg5_25" )
	OPTION="${OPTION} --dataset CIFAR10AMG --noise_scale 5 --multi_noise_scale 25"
	PRIOR=multiplicative_gaussian ;;
    "cifar10amg25_25" )
	OPTION="${OPTION} --dataset CIFAR10AMG --noise_scale 25 --multi_noise_scale 25"
	PRIOR=multiplicative_gaussian ;;
    "cifar10p30" )
	OPTION="${OPTION} --dataset CIFAR10P --noise_lam 30 --blurvh"
	PRIOR=poisson ;;
    "cifar10p10-50" )
	OPTION="${OPTION} --dataset CIFAR10P --noise_lam 10 --noise_lam_high 50 --blurvh"
	PRIOR=poisson ;;
    "cifar10pg30_5" )
	OPTION="${OPTION} --dataset CIFAR10PG --noise_lam 30 --noise_scale 5 --blurvh"
	PRIOR=poisson ;;
    "cifar10pg30_25" )
	OPTION="${OPTION} --dataset CIFAR10PG --noise_lam 30 --noise_scale 25 --blurvh"
	PRIOR=poisson ;;
    * )
	echo "unknown DATASET: ${DATASET}"
	exit 1 ;;
esac

# Set up model
case ${MODEL} in
    "gan" ) ;;
    "ambientgan" )
	OPTION="${OPTION} --noise_measure" ;;
    "sinrgan1" )
	OPTION="${OPTION} --gn_train --prior additive_gaussian" ;;
    "sinrgan2" )
	OPTION="${OPTION} --gn_train --rotation --channel_shuffle --color_inversion" ;;
    "sdnrgan1" )
	OPTION="${OPTION} --gn_train --prior ${PRIOR}" ;;
    "sdnrgan2" )
	OPTION="${OPTION} --gn_train --implicit --prior additive_gaussian" ;;
    "sdnrgan3" )
	OPTION="${OPTION} --gn_train --implicit --color_inversion" ;;
    * )
	echo "unknown MODEL: ${MODEL}"
	exit 1 ;;
esac

# Run
echo "python train.py --out ${OUT} ${OPTION}"
python train.py --out ${OUT} ${OPTION}
