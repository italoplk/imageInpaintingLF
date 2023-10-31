python train.py --model cnr --epochs 1000 --train-repeats 31 --test-repeats 50 --batch-size 16 --n-crops 4 --save /scratch/output_ImageInpainting/train_cnr --project-name train_cnr --lr-scheduler custom_exp
python train.py --model cnr_unet --epochs 1000 --train-repeats 31 --test-repeats 50 --batch-size 16 --n-crops 4 --save /scratch/output_ImageInpainting/train_cnr_unet --project-name train_cnr_unet --lr-scheduler custom_exp
python train.py --model conv --my-decoder --epochs 1000 --train-repeats 31 --test-repeats 50 --batch-size 16 --n-crops 4 --save /scratch/output_ImageInpainting/train_conv_mydecoder --project-name train_conv_mydecoder --lr-scheduler custom_exp
python train.py --model conv_unet --epochs 1000 --train-repeats 31 --test-repeats 50 --batch-size 16 --n-crops 4 --save /scratch/output_ImageInpainting/train_conv_unet --project-name train_conv_unet --lr-scheduler custom_exp

