## Running the train.py

python train.py 'DATASET PATH'


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
