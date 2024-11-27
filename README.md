# Semantic Segmentation with Attention models for Cancer images
## About this project
### This is a research in development!
---

#### Tools used

- Pytorch
- Python
- Numpy
- Pyplot
- GradCam
- ResNet
- Cuda

---


In this project, I study different attention models for segmentation of cancer images, comparing different models and mixing parameters.
The project also tries to justify the network answers, using GradCamX and other systems to show attentions spots of the segmenter and classifier sections.
Down below I'll run through the model process, section by section:



#### Obtaining and augmenting the dataset
<p align="center">
  <img src = "https://github.com/user-attachments/assets/3101357f-cc25-4f25-aacc-1e1a921c5ece" width = 500>
</p>



The first step is to organize the dataset in:
- Training Images
- Training Masks
- Test Images
- Test Masks
After getting the path to all of these images, the model read the images and augment by flipping, cropping and rotating the images, expanding the number of images in the dataset, after that it sends the images to a new folder, with all the new augmented images, then it creates a <b>train dataloader</b> and a <b>test dataloader</b>.
<p align="center">
  <img src = "https://github.com/user-attachments/assets/935ee36a-2006-4a15-8a6f-4b3da6703f45" width = 500>
  <br>
  <em> Example of the flip augmentation, transforming one image in 4.</em>
</p>

---

#### Setting ResNet and GradCam
Resnet is a classification network, usually used to classify a image betwen X labels. In the dataset _OralEpithelium_, there are 4 labels, **healthy**, **mild**, **moderate** and **severe**, each corresponding to the cancer stage.
The model train ResNet network using the dataloaders, getting around 98% Accuracy through out 20 epochs.
The GradCam is a tool used to get the main activation spots in images, using one of the last layers of the network as a base, or in other words, it is "the points of interest of the model". 
By combining both, the model can use the gradcam to show the places where each class of the ResNet model that activated weights(the places that the model paid attention to).
<p align="center">
  <img src = "https://github.com/user-attachments/assets/69cc834e-9000-4e52-8a8f-51cadb5764be" width = 1000>
</p>

---

#### Model training
The segmentation model chose was U-NET model, that uses encoders and decoders to extract features from the image.
<p align="center">
  <img src = "https://github.com/user-attachments/assets/6ded808e-8f06-45c9-86f0-a20adb5bcbd7" width = 500>
  <br>
  <em> Unet structure, blue is residual blocks, orange is concatenation blocks, A is attention block.</em>
</p>


My model also uses attention blocks, to improve the detail on the U-NET segmentation. In this research, there are tests with:
- **No attention model**
- **Attention Gate (AG)**
- **Global Contextual Transformer (GCT)**
- **Self-Regularization Mechanism (SRM)**
  
After choosing what attention mechanism, number of epochs, batch size and learning rate the model will use, the training starts, using Pytorch checkpoint system to save progress and load if the training is interrupted. During the training, the models saves an image after epoch end, at the end it makes a GIF file to show the training process, as below:
<p align="center">
  <img src = "https://github.com/user-attachments/assets/9502cd6c-d5f9-4c79-b1dd-b69cf186ad9d" width = 1000>
  <br>
  <em> The GradCamResNet doesn't animate because it is from a training done before the UNET training, so it is the result of a trained model about the result of only one class, while U-Net is in training.</em>
</p>

---

#### Evaluating

With the training done, the model evaluates using the <b>Test DataLoader</b>, the measurements used are:

- **Accuracy**: The number of correct predictions out of the total of predictions. Higher means better.

- **Loss**: How well the predictions matched the expected result. Lower means better.

- **Dice**: Calculates the similarity between the intersection of two images. Higher means better.

- **Intersection over Union (IoU)**: It calculates the overlap between the predicted image and the ground truth, but it also takes in consideration the union of the boxes, while Dice only accounts for the intersection.Higher means better.

- **Panoptic Quality (PQ)**: Indicates the performance in segmenting instances accurately. Higher means better.

- **Aggregated Jaccard Index (AJI)**: Indicate the performance of the alignment between predicted and ground truth instance boundaries. Higher means better.

- **Specificity**: Measures the proportion of actual negatives that are correctly identified. Higher means better.

- **Sensitivity**: Measures the proportion of actual positives that are correctly identified. Higher means better.

Here are some results from 2 different databases, **OralEpithelium** and **H&E**:
### Attention!! This is still are not the final numbers, results may change later as the research progresses.
![image](https://github.com/user-attachments/assets/3d3792ea-60e7-4f49-b2ab-8c72406e75bd)
![image](https://github.com/user-attachments/assets/340f10a4-6ee6-4bcb-8aed-94a047e1b03e)
The model also displays some metrics for accuracy and loss:
<p align="center">
  <img src = "https://github.com/user-attachments/assets/a119b786-177a-401f-b13d-8cb1fb8f4244" width = 500>
  <br>
  <em> Metrics for Attention Gate in 50 Epochs with 8 batch size, OralEpithelium database, 256px by 256px images and 0.001 Learning Rate .</em>
</p>

After that, the model segments an image from <b>Test DataLoader</b> and asks for the U-Net model to segment it, after the model emphasize the segmented regions and asks for ResNet to classify the image, as below:
<p align="center">
  <img src = "https://github.com/user-attachments/assets/76412bd7-7394-44f2-9d1a-9f58b1aa7c19" width = 1000>
</p>

At the end, the model compares the True Positives, False Negatives and False Positives from the segmentation, as shown below:


![image](https://github.com/user-attachments/assets/7ddd1a3a-f65c-48f6-a030-6cd8b1344860)








