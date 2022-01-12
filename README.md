## Xview3, 2nd place solution
https://iuu.xview.us/

| test split  | aggregate score |
|-------------|-----------------|
| public      | 0.593           |
| holdout     | 0.604           | 

### Inference
To reproduce the submission results, first you need to install the required packages. 
The easiest way is to use docker to build an image or pull a prebuilt docker image.

#### Prebuilt docker image

One can pull the image from docker hub and use it for inference
``` docker pull selimsefhub/xview3:mse_v2l_v2l_v3m_nf_b7_r34``` 

Inference specification is the same as for [XView reference solution](https://github.com/DIUx-xView/xview3-reference) 
``` 
docker run --shm-size 16G --gpus=1 --mount type=bind,source=/home/xv3data,target=/on-docker/xv3data selimsefhub/xview3:mse_v2l_v2l_v3m_nf_b7_r34 /on-docker/xv3data/ 0157baf3866b2cf9v /on-docker/xv3data/prediction/prediction.csv
```
#### Build from scratch
```
docker build -t xview3 .
```

### Training

For training I used an instance 4xRTX A6000. For GPUs with smaller VRAM you will need to reduce crop sizes in configurations.
As I did not make small tiles of large tiff and used memmap instead fast disks like M.2 (ideally in raid0) should be used.

To reproduce training from scratch: 
1. build docker image as described above
2. run docker image with modified entrypoint, e.g. `docker run --rm --network=host --entrypoint /bin/bash --gpus all --ipc host -v /mnt:/mnt -it xview3:latest`
3. run `./train_all.sh NUM_GPUS DATA_DIR SHORE_DIR VAL_OUT_DIR`, where `DATA_DIR` is the root directory with the dataset, `SHORE_DIR` path to shoreline data for validation set,   `VAL_OUT_DIR` any path where csv prediction will be stored on evaluation phase after each epoch
4. example `./train_all.sh 4 /mnt/md0/datasets/xview3/ /mnt/md0/datasets/xview3/shoreline/validation /mnt/md0/datasets/xview3/oof/`
5. it will overwrite existing weights under `weights` directory in container 


### Solution approach

Maritime object detection can be transformed to a binary segmentation and regressing problem using UNet like convolutional neural networks with the multiple outputs.


![Targets](images/targets.png)

### Model architecture and outputs

Generally I used UNet like encoder-decoder model with the following backbones:
- EfficientNet V2 L - best performing
- EfficientNet V2 M
- EfficientNet B7 
- NFNet L0 (variant implemented by Ross Wightman). Works great with small batches due to absence of BatchNorm layers.
- Resnet34

For the decoder I used standard UNet decoder with nearest upsampling without batch norm. SiLU was used as activation for convolutional layers. 
I used full resolution prediction for the masks. 

#### Detection
 
Centers of objects are predicted as gaussians with sigma=2 pixels. Values are scaled between 0-255. 
Quality of dense gaussians is the most important part to obtain high aggregate score.
During the competition I played with different loss functions with varied success:
- Pure MSE loss - had high precision but low recall which was not good enough for the F1 score
- MAE loss did not produce acceptable results
- Thresholded MSE with sum reduction showed best results. Low value predictions did not play any role for the model's quality, so they are ignored. Though loss weight needed to be tuned properly.
 
#### Vessel classification

Vessel masks were prepared as binary round objects with fixed radius (4 pixels)
Missing vessel value was transformed to 255 mask that was ignored in the loss function.
As a loss function I used combination of BCE, Focal and SoftDice losses.

#### Fishing classification
Fishing masks were prepared the same way as vessel masks

#### Length estimation
Length mask - round objects with fixed radius and pixel values were set to length of the object.
Missing length was ignored in the loss function.
As a loss function for length at first I used MSE but then change to the loss function that directly reflected the metric.
I.e.`length_loss = abs(target - predicted_value)/target`


