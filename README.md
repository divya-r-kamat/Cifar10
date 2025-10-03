##  CIFAR-10 
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes. There are 50000 training images and 10000 test images.

<img width="901" height="557" alt="image" src="https://github.com/user-attachments/assets/a1bd84dc-7663-402b-ba7d-31f7dc6614be" />


### Objective
- has the architecture to C1C2C3C40 (No MaxPooling, but convolutions, where the last one has a stride of 2 instead) (NO restriction on using 1x1) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
- total RF must be more than 44
- One of the layers must use Depthwise Separable Convolution
- One of the layers must use Dilated Convolution
- use GAP (compulsory):- add FC after GAP to target #of classes (optional)
- Use the albumentation library and apply:
    - horizontal flip
    - shiftScaleRotate
    - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
- achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.



### Model1

### Target:
- Get the set-up right
- Set Transforms
- Set Data Loader
- Set Basic Working Code
- Set Basic Training & Test Loop
- Get the basic skeleton right, with 4 Convolution blocks , No maxpooling and receptive field of > 44

### Results:
- Parameters: 1,605,600
- Best Training Accuracy: 99.9%
- Best Test Accuracy: 80%

### Analysis
- Model is clearly overfitting
- Also model parameters can be reduced further

## Model2

### Target:
- Reduce the model parameters by reducing the number of channels and Added depth wise convolution
- added dialated convolution for increased receptive field
  
### Results:
- Parameters: 97,264
- Best Training Accuracy: 72%
- Best Test Accuracy: 67%

### Analysis
- Train/test accuracy reduced after reducing the number of parameters
- still see some overfitting

## Model3

### Target:
- Added data augmentation - HorizontalFlip, ShiftScaleRotate, ColorJitter, CoarseDropout (Cutout)

### Results:
- Parameters: 97,264
- Best Training Accuracy: 84%
- Best Test Accuracy: 85.4%
- Epochs Run: 50
- 
### Analysis
- Starting performance:
    - Epoch 1 test accuracy = 47.8%, which is higher than raw baselines (~35–40%) → augmentations helped the model generalize even in the very first epoch.
- Steady improvements:

    - By Epoch 10: ~80% accuracy.
    - By Epoch 20: ~81–82%, but with some plateauing.
    - By Epoch 30–40: Accuracy stabilized around 83–85%

- helped prevent overfitting (train accuracy was high, but test accuracy tracked well, no big gap)
- Albumentations helped the model learn more invariant and robust features, leading to higher test accuracy (~85.4%) and stronger generalization compared to training without augmentation

### Model Architecture

    CIFAR10(
      (conv1): Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ReLU()
        (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Dropout2d(p=0.01, inplace=False)
        (4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): ReLU()
        (6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): Dropout2d(p=0.01, inplace=False)
      )
      (trans1): Sequential(
        (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(2, 2))
        (1): ReLU()
      )
      (conv2): Sequential(
        (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ReLU()
        (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Dropout2d(p=0.01, inplace=False)
        (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (5): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
        (6): ReLU()
        (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): Dropout2d(p=0.01, inplace=False)
      )
      (trans2): Sequential(
        (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(2, 2))
        (1): ReLU()
      )
      (conv3): Sequential(
        (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(2, 2), bias=False)
        (1): ReLU()
        (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Dropout2d(p=0.01, inplace=False)
        (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): ReLU()
        (6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): Dropout2d(p=0.01, inplace=False)
      )
      (trans3): Sequential(
        (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(2, 2))
        (1): ReLU()
      )
      (conv4): Sequential(
        (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ReLU()
        (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Dropout2d(p=0.01, inplace=False)
        (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (5): Conv2d(32, 10, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (gap): Sequential(
        (0): AdaptiveAvgPool2d(output_size=1)
      )
    )

## Receptive Field
| Operation      | nin | Channels | Output\_Channels | padding | kernel | stride | dilation | nout | jin | jout | rin | rout |
| -------------- | --- | -------- | ---------------- | ------- | ------ | ------ | -------- | ---- | --- | ---- | --- | ---- |
| Conv-B1 | 32  | 3        | 32               | 1       | 3      | 1      | 1        | 32   | 1   | 1    | 1   | 3    |
| Conv-B1 | 32  | 32       | 64               | 1       | 3      | 1      | 1        | 32   | 1   | 1    | 3   | 5    |
| Conv-TB | 32  | 64       | 32               | 1       | 1      | 2      | 1        | 16   | 1   | 2    | 5   | 5    |
| Conv-B2 | 16  | 32       | 32               | 1       | 3      | 1      | 1        | 16   | 2   | 2    | 5   | 9    |
| Conv-B2 | 16  | 32       | 32               | 1       | 3      | 1      | 1        | 16   | 2   | 2    | 9   | 13   |
| Conv-B2 | 16  | 32       | 32               | 1       | 1      | 1      | 1        | 18   | 2   | 2    | 13  | 13   |
| Conv-TB | 18  | 32       | 64               | 1       | 1      | 2      | 1        | 9    | 2   | 4    | 13  | 13   |
| Conv-B3 | 9   | 64       | 32               | 1       | 3      | 1      | 2        | 7    | 4   | 8    | 13  | 29   |
| Conv-B3 | 7   | 32       | 64               | 1       | 3      | 1      | 1        | 7    | 8   | 8    | 29  | 45   |
| Conv-TB | 7   | 64       | 64               | 1       | 3      | 2      | 1        | 4    | 8   | 16   | 45  | 61   |
| Conv-B4 | 4   | 64       | 16               | 1       | 3      | 1      | 1        | 4    | 16  | 16   | 61  | 93   |
| Conv-B4 | 4   | 16       | 32               | 1       | 3      | 1      | 1        | 4    | 16  | 16   | 93  | 125  |
| Conv-B4 | 4   | 32       | 32               | 1       | 1      | 1      | 1        | 6    | 16  | 16   | 125 | 125  |

## Train and Test Logs

    Epoch 1
    Train loss=1.3733 batch_id=390 Accuracy=36.28: 100% 391/391 [00:22<00:00, 17.39it/s]
    Test set: Average loss: 1.4182, Accuracy: 4784/10000 (47.84%)
    
    Epoch 2
    Train loss=1.0619 batch_id=390 Accuracy=53.56: 100% 391/391 [00:21<00:00, 17.82it/s]
    Test set: Average loss: 1.1051, Accuracy: 6072/10000 (60.72%)
    
    Epoch 3
    Train loss=1.0864 batch_id=390 Accuracy=61.09: 100% 391/391 [00:22<00:00, 17.69it/s]
    Test set: Average loss: 0.9158, Accuracy: 6668/10000 (66.68%)
    
    Epoch 4
    Train loss=1.0844 batch_id=390 Accuracy=66.29: 100% 391/391 [00:21<00:00, 18.30it/s]
    Test set: Average loss: 0.8054, Accuracy: 7171/10000 (71.71%)
    
    Epoch 5
    Train loss=0.8940 batch_id=390 Accuracy=68.69: 100% 391/391 [00:21<00:00, 18.57it/s]
    Test set: Average loss: 0.7619, Accuracy: 7283/10000 (72.83%)
    
    Epoch 6
    Train loss=0.8382 batch_id=390 Accuracy=70.76: 100% 391/391 [00:21<00:00, 17.96it/s]
    Test set: Average loss: 0.7019, Accuracy: 7511/10000 (75.11%)
    
    Epoch 7
    Train loss=0.8772 batch_id=390 Accuracy=72.33: 100% 391/391 [00:23<00:00, 16.60it/s]
    Test set: Average loss: 0.6704, Accuracy: 7717/10000 (77.17%)
    
    Epoch 8
    Train loss=0.7961 batch_id=390 Accuracy=73.77: 100% 391/391 [00:22<00:00, 17.39it/s]
    Test set: Average loss: 0.6437, Accuracy: 7815/10000 (78.15%)
    
    Epoch 9
    Train loss=0.6662 batch_id=390 Accuracy=74.90: 100% 391/391 [00:22<00:00, 17.49it/s]
    Test set: Average loss: 0.6353, Accuracy: 7834/10000 (78.34%)
    
    Epoch 10
    Train loss=0.7073 batch_id=390 Accuracy=75.83: 100% 391/391 [00:21<00:00, 18.02it/s]
    Test set: Average loss: 0.5903, Accuracy: 7988/10000 (79.88%)
    
    Epoch 11
    Train loss=0.6463 batch_id=390 Accuracy=76.81: 100% 391/391 [00:21<00:00, 18.59it/s]
    Test set: Average loss: 0.5772, Accuracy: 8076/10000 (80.76%)
    
    Epoch 12
    Train loss=0.7271 batch_id=390 Accuracy=77.64: 100% 391/391 [00:21<00:00, 17.98it/s]
    Test set: Average loss: 0.5590, Accuracy: 8122/10000 (81.22%)
    
    Epoch 13
    Train loss=0.6748 batch_id=390 Accuracy=78.26: 100% 391/391 [00:22<00:00, 17.59it/s]
    Test set: Average loss: 0.5457, Accuracy: 8127/10000 (81.27%)
    
    Epoch 14
    Train loss=0.4418 batch_id=390 Accuracy=78.80: 100% 391/391 [00:22<00:00, 17.56it/s]
    Test set: Average loss: 0.5363, Accuracy: 8167/10000 (81.67%)
    
    Epoch 15
    Train loss=0.5677 batch_id=390 Accuracy=79.07: 100% 391/391 [00:22<00:00, 17.70it/s]
    Test set: Average loss: 0.5321, Accuracy: 8195/10000 (81.95%)
    
    Epoch 16
    Train loss=0.5727 batch_id=390 Accuracy=79.01: 100% 391/391 [00:21<00:00, 18.04it/s]
    Test set: Average loss: 0.5318, Accuracy: 8206/10000 (82.06%)
    
    Epoch 17
    Train loss=0.5811 batch_id=390 Accuracy=79.22: 100% 391/391 [00:22<00:00, 17.44it/s]
    Test set: Average loss: 0.5293, Accuracy: 8211/10000 (82.11%)
    
    Epoch 18
    Train loss=0.5760 batch_id=390 Accuracy=79.34: 100% 391/391 [00:21<00:00, 17.78it/s]
    Test set: Average loss: 0.5292, Accuracy: 8201/10000 (82.01%)
    
    Epoch 19
    Train loss=0.7857 batch_id=390 Accuracy=78.85: 100% 391/391 [00:22<00:00, 17.48it/s]
    Test set: Average loss: 0.5318, Accuracy: 8187/10000 (81.87%)
    
    Epoch 20
    Train loss=0.5863 batch_id=390 Accuracy=78.47: 100% 391/391 [00:22<00:00, 17.46it/s]
    Test set: Average loss: 0.5327, Accuracy: 8185/10000 (81.85%)
    
    Epoch 21
    Train loss=0.6139 batch_id=390 Accuracy=78.35: 100% 391/391 [00:22<00:00, 17.61it/s]
    Test set: Average loss: 0.5578, Accuracy: 8110/10000 (81.10%)
    
    Epoch 22
    Train loss=0.6776 batch_id=390 Accuracy=77.77: 100% 391/391 [00:21<00:00, 17.86it/s]
    Test set: Average loss: 0.5538, Accuracy: 8118/10000 (81.18%)
    
    Epoch 23
    Train loss=0.7160 batch_id=390 Accuracy=77.67: 100% 391/391 [00:21<00:00, 18.51it/s]
    Test set: Average loss: 0.5677, Accuracy: 8045/10000 (80.45%)
    
    Epoch 24
    Train loss=0.5566 batch_id=390 Accuracy=76.97: 100% 391/391 [00:21<00:00, 17.99it/s]
    Test set: Average loss: 0.5704, Accuracy: 8066/10000 (80.66%)
    
    Epoch 25
    Train loss=0.8247 batch_id=390 Accuracy=76.68: 100% 391/391 [00:22<00:00, 17.61it/s]
    Test set: Average loss: 0.5813, Accuracy: 8003/10000 (80.03%)
    
    Epoch 26
    Train loss=0.5705 batch_id=390 Accuracy=76.84: 100% 391/391 [00:22<00:00, 17.67it/s]
    Test set: Average loss: 0.5775, Accuracy: 8025/10000 (80.25%)
    
    Epoch 27
    Train loss=0.7557 batch_id=390 Accuracy=76.71: 100% 391/391 [00:23<00:00, 16.61it/s]
    Test set: Average loss: 0.5887, Accuracy: 7989/10000 (79.89%)
    
    Epoch 28
    Train loss=0.5942 batch_id=390 Accuracy=76.94: 100% 391/391 [00:22<00:00, 17.46it/s]
    Test set: Average loss: 0.5692, Accuracy: 8070/10000 (80.70%)
    
    Epoch 29
    Train loss=0.6342 batch_id=390 Accuracy=77.12: 100% 391/391 [00:22<00:00, 17.52it/s]
    Test set: Average loss: 0.5850, Accuracy: 8016/10000 (80.16%)
    
    Epoch 30
    Train loss=0.5869 batch_id=390 Accuracy=77.56: 100% 391/391 [00:21<00:00, 18.22it/s]
    Test set: Average loss: 0.5736, Accuracy: 8017/10000 (80.17%)
    
    Epoch 31
    Train loss=0.5510 batch_id=390 Accuracy=77.75: 100% 391/391 [00:21<00:00, 18.37it/s]
    Test set: Average loss: 0.6148, Accuracy: 7871/10000 (78.71%)
    
    Epoch 32
    Train loss=0.7769 batch_id=390 Accuracy=78.21: 100% 391/391 [00:21<00:00, 18.13it/s]
    Test set: Average loss: 0.5778, Accuracy: 8027/10000 (80.27%)
    
    Epoch 33
    Train loss=0.5935 batch_id=390 Accuracy=78.64: 100% 391/391 [00:22<00:00, 17.75it/s]
    Test set: Average loss: 0.5394, Accuracy: 8149/10000 (81.49%)
    
    Epoch 34
    Train loss=0.4707 batch_id=390 Accuracy=78.75: 100% 391/391 [00:22<00:00, 17.59it/s]
    Test set: Average loss: 0.5328, Accuracy: 8207/10000 (82.07%)
    
    Epoch 35
    Train loss=0.7606 batch_id=390 Accuracy=79.65: 100% 391/391 [00:22<00:00, 17.66it/s]
    Test set: Average loss: 0.5223, Accuracy: 8252/10000 (82.52%)
    
    Epoch 36
    Train loss=0.7202 batch_id=390 Accuracy=79.90: 100% 391/391 [00:22<00:00, 17.57it/s]
    Test set: Average loss: 0.4979, Accuracy: 8312/10000 (83.12%)
    
    Epoch 37
    Train loss=0.6107 batch_id=390 Accuracy=80.66: 100% 391/391 [00:21<00:00, 18.23it/s]
    Test set: Average loss: 0.5043, Accuracy: 8265/10000 (82.65%)
    
    Epoch 38
    Train loss=0.4896 batch_id=390 Accuracy=81.12: 100% 391/391 [00:22<00:00, 17.55it/s]
    Test set: Average loss: 0.4840, Accuracy: 8347/10000 (83.47%)
    
    Epoch 39
    Train loss=0.4407 batch_id=390 Accuracy=81.73: 100% 391/391 [00:21<00:00, 18.31it/s]
    Test set: Average loss: 0.4733, Accuracy: 8363/10000 (83.63%)
    
    Epoch 40
    Train loss=0.4378 batch_id=390 Accuracy=82.22: 100% 391/391 [00:22<00:00, 17.74it/s]
    Test set: Average loss: 0.4662, Accuracy: 8391/10000 (83.91%)
    
    Epoch 41
    Train loss=0.5727 batch_id=390 Accuracy=82.78: 100% 391/391 [00:22<00:00, 17.69it/s]
    Test set: Average loss: 0.4471, Accuracy: 8491/10000 (84.91%)
    
    Epoch 42
    Train loss=0.6153 batch_id=390 Accuracy=83.23: 100% 391/391 [00:22<00:00, 17.69it/s]
    Test set: Average loss: 0.4466, Accuracy: 8460/10000 (84.60%)
    
    Epoch 43
    Train loss=0.4507 batch_id=390 Accuracy=83.52: 100% 391/391 [00:22<00:00, 17.72it/s]
    Test set: Average loss: 0.4359, Accuracy: 8500/10000 (85.00%)
    
    Epoch 44
    Train loss=0.5061 batch_id=390 Accuracy=83.97: 100% 391/391 [00:22<00:00, 17.62it/s]
    Test set: Average loss: 0.4360, Accuracy: 8521/10000 (85.21%)
    
    Epoch 45
    Train loss=0.5126 batch_id=390 Accuracy=83.82: 100% 391/391 [00:21<00:00, 18.58it/s]
    Test set: Average loss: 0.4319, Accuracy: 8525/10000 (85.25%)
    
    Epoch 46
    Train loss=0.3960 batch_id=390 Accuracy=84.19: 100% 391/391 [00:20<00:00, 18.69it/s]
    Test set: Average loss: 0.4312, Accuracy: 8522/10000 (85.22%)
    
    Epoch 47
    Train loss=0.5862 batch_id=390 Accuracy=84.05: 100% 391/391 [00:22<00:00, 17.65it/s]
    Test set: Average loss: 0.4312, Accuracy: 8533/10000 (85.33%)
    
    Epoch 48
    Train loss=0.4563 batch_id=390 Accuracy=84.02: 100% 391/391 [00:23<00:00, 16.92it/s]
    Test set: Average loss: 0.4313, Accuracy: 8550/10000 (85.50%)
    
    Epoch 49
    Train loss=0.5708 batch_id=390 Accuracy=83.85: 100% 391/391 [00:22<00:00, 17.69it/s]
    Test set: Average loss: 0.4288, Accuracy: 8539/10000 (85.39%)
    
    Epoch 50
    Train loss=0.6981 batch_id=390 Accuracy=83.74: 100% 391/391 [00:21<00:00, 17.82it/s]
    Test set: Average loss: 0.4348, Accuracy: 8540/10000 (85.40%)

## Validation plot

<img width="1045" height="700" alt="image" src="https://github.com/user-attachments/assets/6d5141b9-acfb-44ea-859a-78b53070ffdc" />

## Misclassified Images

<img width="677" height="690" alt="image" src="https://github.com/user-attachments/assets/d58df419-8480-487b-b8b6-21241f52bee4" />
