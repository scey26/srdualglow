# Update log

<07/10/2021(by Changyeop)>

DONE

1) Removing conditional network (CN)
2) Temperature should be set 0 to preserve contents

TODO

1) Debugging the right glow for convergence
2) Adding skip-connection to use LR images
3) Not necessary to be bound by injector 
4) Due to the GPU memory, let's set n_flow=16 n_block=2 following the SRFlow

Before works
  
1) Reproducing the FullGlow network
2) Training left glow first, and then training right glow
3) Adding transition layers referring to SRFlow (proved by training the left glow model -> work!)
4) Adding the affine injector reffering to SRFlow (not converge)


# Train command

## Training the left glow
```bash
python train_onlyleft.py 'DATASET PATH'  --save_folder 'SAVING FOLDER NAME' --batch 'BATCH SIZE' --n_flow 'Num FLOWS' --n_block 'Num BLOCKS'
```
ex) python train_freezeleft.py './dataset' --save_folder 210710_test2 --batch 4 --n_flow 16 --n_block 2

## Training the right glow
```bash
python train_freezeleft.py 'DATASET PATH'  --save_folder 'SAVING FOLDER NAME' --left_glow_params 'LEFT GLOW MODEL PATH' --temp 'TEMP' --batch 'BATCH' --n_flow 'Num FLOWS' --n_block 'Num BLOCKS'
```
ex) python train_freezeleft.py './dataset' --save_folder 210710_test2 --left_glow_params '' --temp 0 --batch 4 --n_flow 16 --n_block 2

# Recording areas

### Get started

First of all, we will train the left Glow model without conditioning. You can train with `python train_onlyleft.py PATH_FOR_CELEBA --save_folder SAVE_PATH`.
Even with 200 iterations, reconstruction with encoded z from input images works well.
<p align="center">
<img width="70%" src="src/train_onlyleft_images/gen_lr_000201.png">
</p>

With default hyperparameters, generated images from random z at several training iterations are as follows (really subjective).

<p align="center">
<img width="70%" src="src/train_onlyleft_images/gen_lr_randz_004001.png">
</p>

At 4,000 iterations, faces are coming.

<p align="center">
<img width="70%" src="src/train_onlyleft_images/gen_lr_randz_015001.png">
</p>

At 15,000 iterations, eyes are coming.

<p align="center">
<img width="70%" src="src/train_onlyleft_images/gen_lr_randz_030001.png">
</p>

At 30,000 iterations, eyes and fine details are coming.

<p align="center">
<img width="70%" src="src/train_onlyleft_images/gen_lr_randz_107001.png">
</p>

At 107,000 iterations..........
