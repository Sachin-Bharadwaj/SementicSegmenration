# SementicSegmentation & Depth Estimation
# DeepLabV3 trained on Synscapes dataset for more than 24 hours with resnet 18 backbone with dilation in last 2 layers. Key features of architecture are <br>
1. Dilations to increase receptive field even further <br>
2. Spatial pyramid pooling <br>
3. Fusing features from tlow level in encoder-decoder style to get better resolution <br>
4. Seperate decoder with Multi task learning <br>

Here is the performance. mIOU > 84% on 19 classes. <br>
<img width="740" alt="image" src="https://github.com/Sachin-Bharadwaj/SementicSegmentation/assets/26499326/5959d829-e5e6-4c72-93ab-5aab6c6d1ee0">


Validation loss: <br>
 <img width="1170" alt="image" src="https://github.com/Sachin-Bharadwaj/SementicSegmentation/assets/26499326/07189c6b-12d5-453b-bf7d-20a2ae218a8e">
 <br>

mIOU on val set: <br>
 <img width="1171" alt="image" src="https://github.com/Sachin-Bharadwaj/SementicSegmentation/assets/26499326/63473ee5-e7b9-4237-a1ce-7c2d3ea632ca">
 <br>


