python train.py --model cnr --epochs 2 --train-repeats 31 --test-repeats 50 --batch-size 16 --n-crops 4 --save /scratch/output_ImageInpainting/delete_train_cnr --project-name delete_train_cnr --lr-scheduler custom_exp
python train.py --model cnr_unet --epochs 2 --train-repeats 31 --test-repeats 50 --batch-size 16 --n-crops 4 --save /scratch/output_ImageInpainting/delete_train_cnr_unet --project-name delete_train_cnr_unet --lr-scheduler custom_exp
python train.py --model conv --my-decoder --epochs 2 --train-repeats 31 --test-repeats 50 --batch-size 16 --n-crops 4 --save /scratch/output_ImageInpainting/delete_train_conv_mydecoder --project-name delete_train_conv_mydecoder --lr-scheduler custom_exp
python train.py --model conv_unet --epochs 2 --train-repeats 31 --test-repeats 50 --batch-size 16 --n-crops 4 --save /scratch/output_ImageInpainting/delete_train_conv_unet --project-name delete_train_conv_unet --lr-scheduler custom_exp

