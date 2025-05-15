#!/bin/bash
#SBATCH --job-name=matterport
#SBATCH --output=sbatch_log/matterport_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu06,bmicgpu07,bmicgpu08,bmicgpu09,bmicgpu10,octopus01,octopus02,octopus03,octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qi.ma@vision.ee.ethz.ch



# conda create -n feature_3dgs python=3.9 -y
source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh

conda activate feature_3dgs

export CONDA_OVERRIDE_CUDA=11.8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

cd /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/encoders/lseg_encoder
python -u encode_images_holicity.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir /srv/beegfs02/scratch/qimaqi_data/data/holicity_val_set_suite/original_data/ --test-rgb-dir /srv/beegfs02/scratch/qimaqi_data/data/holicity_val_set_suite/original_data/ytwUEEljP6RgoV0MviqvsQ_LD/image --workers 0
