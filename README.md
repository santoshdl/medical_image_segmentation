# Medical Image Segmentation
The resolution of image is very high ! 


## Mask Preparation

### Why 512 x 512 ?

Reasonable size which covered the particles in full.  We could go for a higher resolution to increase receptive field and accuracy.


## Removing patches carrying no information
Done this to reduce dataset size. There were many masked patches with only background.

## Trained Model weights
https://drive.google.com/file/d/1QuuNhgFUcbqmEhHIGVKbuOuWnUSf-f9L/view?usp=sharing

## Result
<br>
<img width="100%" src="predictions.png"></a>
<br>
<br>
## References
Understanding important features of deep learning models for segmentation of high-resolution transmission electron microscopy images
https://rdcu.be/c8YtO

https://github.com/nikhilroxtomar/RGB-Mask-to-Single-Channel-Mask-for-Multiclass-Segmentation/blob/main/rgb_mask_to_single_channel_mask.py

Architecture:
https://www.kaggle.com/code/vikram12301/multiclass-semantic-segmentation-pytorch

Image patching : https://levelup.gitconnected.com/how-to-split-an-image-into-patches-with-python-e1cf42cf4f77

## Issues
DecompressionBombError: Image size (242221056 pixels) exceeds limit of 178956970 pixels, could be decompression bomb DOS attack.

https://github.com/Belval/pdf2image/issues/76


Unable to find a valid cuDNN algorithm to run convolution

https://stackoverflow.com/questions/61467751/unable-to-find-a-valid-cudnn-algorithm-to-run-convolution