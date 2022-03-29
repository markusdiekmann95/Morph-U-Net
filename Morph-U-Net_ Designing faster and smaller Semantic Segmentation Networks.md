# Morph-U-Net: Designing faster and smaller Semantic Segmentation Networks
###### Author: `Markus Diekmann`
###### University:     `Westfälische Wilhelms-Universität Münster`
###### Study Program:  `Information Systems MSc.`
###### Seminar:        `Recent Trends in Deep Learning`
###### [Github Repository](https://github.com/markusdiekmann95/Morph-U-Net)

##TODO: Adjust the formulas to fit into the github markdown

## Introduction
In the field of machine learning and deep learning development, the focus is usually on high-performance solutions to maximize performance **[1]**. However, resources regarding to hardware constraints were disregarded and the focus was mostly on cloud-based solutions, for example. Nevertheless, the models should also be able to run on small, resource-constrained systems like microcontrollers **[2]**. TinyML addresses this bottleneck and aims to run high-performance models on low latency, low power, and low bandwidth devices **[3]**. Specific packages have already been developed for this, such as TensorFlow Light. Alternatively, models can be optimized depending on available resources **[4]**. Resources that can be optimized in this context are for example FLOPs, model size, or latency. 

In this blog, the emphasis is on optimizing deep learning architectures with a focus on model size without taking a big hit on performance. The U-Net for semantic segmentation is taken into account as the architecture to be optimized. To achieve this goal, the background of semantic segmentation as well as model optimization is investigated first. Then, MorphNet is presented, which is a promising technique for the optimization of neural network architectures. To assess the feasibility of applying MorphNet to U-Net, an experimental study is conducted. In the course of this, different regularization techniques will be investigated and the preservation of symmetry for the U-Net will be discussed to assess the importance of the U-Net's symmetry.

## Background

### U-Net
Convolutional neural networks typical use is for classification where the whole image is assigned to a class. This enables us to know what is on the image. However, sometimes it is not sufficient just to know what is on the image, but also where it is on the image **[5]**. Thus, it is useful to assign each pixel of the image to a specific class. 
> ![](https://radosgw.public.os.wwu.de/pad/uploads/upload_38b11cd5ae88387b28585705e8110a2b.png)
> Figure 1: Segmentation of Buildings

This is possible for example via a sliding-window where the network iterates over the whole image and creates a prediction based on a local region around that image. A better solution for this is the U-Net **[5]**, which takes the whole image as input and gives a segmentation mask where each pixel is assigned to a class as output. 
> ![](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)
> Figure 2: U-Net Architecture **[5]**

The U-Net is composed of two paths: the contracting path on the left and the expansive path on the right. When considering the contracting path, it can be seen that it is a typical convolutional network that takes the input image as input and has multiple layers with a double convolution block with ReLU as activation function after each convolution. In addition, a max-pooling operation for downsampling is performed after each of double convolution block. This leads to the downsizing of the image. Additionally, the number of feature maps is doubled after each downsampling. The expansive path also consists of multiple layers with double convolution blocks. But instead of max pooling operations, up-convolutions are applied. Each double convolution block receives the output from the previous up-convolution and the cropping from the contracting path as input which will be concatenated and go through the double convolutions. The cropping is performed due to the loss of border pixels resulting from the convolutions. During the expansive path, the number of feature maps will be reduced and the image size is upsized again. In summary, the contracting path is responsible to detect what is on the image and the expansive path ensures the retrieval of spatial information. 

However, the U-Net is computationally expensive and consists of millions of parameters. It is constructed often in his baseline architecture with doubling and halving the number of feature maps. It is possible that many feature maps are not useful at all and thus just waste resources. Therefore, we need to find a way to remove them!


### Network Shrinking
For shrinking a network there are multiple possible approaches. The first approach would be the *width multiplier* $w$ where all layer sizes were multiplied with a defined factor **[6]**. The factor must be $w>0$ and if it is $w>1$ it will expand the network and with $w<1$ the network will be shrink. Let us assume that we have a convolutional neural network $N$ with the convolution layer sizes $[64,128,256,512]$ and we want to apply $w$ on $N$. If we want to decrease our network by $25\%$, we set $w$ to $0.75$. The computation and the new network will look as follows: $[0.75*64,0.75*128,0.75*256,0.75*512] = [48,96,192,384]$. To find our optimal width multiplier, we must set constraints. E.g., we set a maximum $C$ for the model size $F$. In form of pseudocode it could be performed like this:

1. Find the largest $w$ such that $F(w * N) \le C$
2. Return $w * N$

This approach is very simple to apply but suffers from decreased quality of the network design.

The second approach, to be considered, is *sparsifying regularization* **[6]** where the loss function $L$ is extended by costs for the neurons in form of a penalty $G$. 
This would look like: \begin{aligned}\theta^* = argmin\{L(\theta)+\lambda*G(\theta)\}\end{aligned}. Inefficient weights of neurons will be decreased and this could be used to have new layer output widths. An advantage in comparison to the *width multiplier*, this approach could change the relative layer sizes. However, the satisfaction of the constraint is not guaranteed and the performance could become worse significantly.

Both approaches have their advantages and drawbacks. In the following the *MorphNet* technique will be investigated which integrates both approaches with each other.


## MorphNet
MorphNet was introduced by Gordon et al. **[6]** and combines the two approaches *width multiplier* and *sparsifying regularization* with each other. In addition, the constraint of the selected resource will be considered in the MorphNet. In the following, the MorphNet technique will be explained step by step **[6]**. The MorphNet approach is algorithm-based and works as follows:
1. Train network with $\theta^∗  = argmin\{L(\theta)+λ∗G(\theta)\}$
2. Identify new widths for each layer
3. Apply *width multiplier* with largest w such that $F(w * N)  \leq C$
4. Repeat steps 1-3 until satisfying network
5. Return network

### Network Shrinking
The first two steps of the MorphNet algorithm are responsible for this stage. For shrinking the network **[7]**, *sparsifying regularization* $G(θ)$ will be applied. However, $G(θ)$ is not only basic regularization, it also considers the targeted resource and if a neuron is alive or not. There are many possibilities for the *targeted resource*: e.g., FLOPs per inference, model size, or latency. In this study, we will focus on the model size of the network. The model size is defined by the number of parameters in a network and due to the convolutions it will be computed by matrix multiplications **[8]**. The goal is to reduce the number of parameters in the model what can be done by reducing the number of feature maps in a convolutional neural network. For example, we receive $64$ feature channels as input and perform convolutions with our $3 * 3$ kernel on it and have an output of $128$ feature channels. The number of parameters would be $64*128*9+128=73856$. If we can reduce the output feature maps to $100$ the number of parameters in the model will be decreased as can be seen in the following computation: $64*100*9+100=57700$. The model size target $C$ for a specific layer can be computed as follows: $C = filter\_height * filter\_width$. For example, we have a $3*3$ kernel for our convolution layer which results into $C = 9$. It can be implemented very easily. 
```
def model_size_C(layer):
  '''Takes a layer as input and compute it's model size C '''
  res = torch.tensor(0.)
  x = torch.Tensor(2)
  x[0] = torch.tensor([layer.shape[2]])
  x[1] = torch.tensor([layer.shape[3]])
  res = torch.prod(x)
  return res.cuda()
  ```
To consider if a neuron is alive or not, an indicator function will be applied. Thereby, the indicator function returns $1$ if the weight of a neuron meets a defined threshold and if not it returns $0$. This will be computed once for all input neurons and once for all output neurons as follows for a specific layer. 
```
def zero_out_indicator(layer, threshold=0.0001):
  '''
  layer: Convolution Layer Weights
  threshold: If neuron is below this threshold, it is not alive
  returns: Number of neurons which are alive
  '''
  res = torch.Tensor(layer.size())
  al = 0
  res = torch.where(layer.abs() > torch.tensor(threshold).cuda(), torch.tensor(1.).cuda(), torch.tensor(0.).cuda())
  al = torch.sum(res)
  return al
  ```
The function investigates each weight in a given layer if its absolute value is below the threshold or not and computes, in the end, the number of weights being alive. The higher the threshold is, the more weights will be declared as not alive. Thus, a suitable threshold must be selected. An example would be that our convolution layer has the following 8 weights after applying sparsifying regularization: $[0.2,0.0015,-0.04,0.01,0.00001,0.13,-0.0005,0.02]$ and our initiated threshold is $0.001$. The fifth and the seventh weight are below the threshold and all other weights have higher values. So, $6$ out of $8$ weights will be seen as alive and the indicator function will return $6$.

In addition, there is the regularization itself, which will be discussed in more detail later in the blog post and will therefore be referred to as $R$ in the following. All three parts together will result in the following term for a specific layer **[6]**. Thereby for a specific layer, the input and the output weights are considered.
\begin{aligned}G(\theta, layer L) = C * R_{inputs}*\sum Alive_{outputs} + C *R_{outputs}*\sum Alive_{inputs}\end{aligned} This term is added to the loss function and penalizes inefficient weights and this finishes the first step of the MorphNet algorithm. Before adding the term to the loss function, a value for $\lambda$ must be defined as the regularization strength. The higher $\lambda$, the stronger the regularization will be and more neurons will be zeroed out. Thus, a suitable $\lambda$ must be selected. In the second step, the task is to identify the new widths of the shrunk layers. For this purpose, it is investigated how many neurons are still alive after sparsifying regularization. 
```
def percentage_waste_conv(layer, threshold=0.0001):
  '''
  @layer: weights of a layer
  @threshold: all absolute values of weights below this threshold are zero
  @return: number of dead weights in percentage for a layer
  '''
  w = layer
  sparsified_w = sparsify(w, threshold)
  if sparsified_w.max() == 0:
    return 1.
  non_sparse_w = torch.nonzero(torch.flatten(sparsified_w))
  return 1-(non_sparse_w.numel()/(w.numel()+1e-10))
  
def sparsify(weights, threshold=0.0001):
  '''
  @weights: all weights of a layer
  @threshold: all absolute values of weights below this threshold are zero
  @return: weights where dead weights are zero
  '''
  w = weights.cpu()
  res = w.where(w.abs() > torch.tensor(threshold).cpu(), torch.tensor(0.).cpu())
  return res
  ```
To receive the new width of a layer, the dead weights are initially set to $0$ by the function $sparsify$ and then it will be computed how many weights are zeroed out in percentage by the function $percentage\_waste$ and for receiving the new width the old width will be reduced by this value. An example would be that we have our initial network with these widths: $[64,128,256,512]$ and our shrinking results into these values of waste for each layer $[0.0625, 0.125, 0.2578, 0.3359]$. After reducing our initial width by these values, we receive our new widths $[60, 111, 190, 340]$ in form of a shrunk network.

### Network Expanding

After shrinking the network, we have reduced our targeted resource. However, the performance suffers usually. This is the reason why we apply the width multiplier uniformly on all layers **[6]**. When choosing the value of the width multiplier, it is important that the constraint of the targeted resource is not violated. As there is nothing new to the *width multiplier*, no further explanations are necessary. The value of the width multiplier should be $>1$ as we want to increase the network. Applying it with $w=1.2$ on our example from the previous step will result into $[1.2 * 60, 1.2 * 111, 1.2* 190, 1.2* 340]$ = $[72,133,228,408]$ as our expanded network under constraints of model size **[7]**. As can be seen, the layers that have been reduced less before are now larger than initially. Layers that have been reduced more remain smaller than initially.

## Morph-U-Net

MorphNet was already applied on modern architectures for classification tasks like Inception, MobileNet, or ResNet **[6]**. Up to now, the focus was not to apply it on semantic segmentation tasks. Therefore, we will investigate now, if the MorphNet approach is applicable to semantic segmentation networks like U-Net. Therefore, we will combine the two topics MorphNet and U-Net with each other. 

### Experimental Study Setup

In order to test the feasibility, an accompanying experimental study will be carried out. For this study, the Inria Aerial Image Labeling Data Set **[9]** will be used. As the dataset is very large and we have resource constraints, only a subset with smaller image sizes of it will be used. Therefore, the city Vienna was selected and we use $600$ images with $500*500$ pixels from it with RGB color channel. The data set contains satellite images and corresponding building masks to solve binary pixel-wise classification tasks as semantic segmentation. The Figure 1 above is an example from this data set.

As our seed network, we will use the origin U-Net architecture for Biomedical Image Segmentation with padding and *Batch Normalization*. Our optimizer will be ADAM and [Dice Loss](https://arxiv.org/pdf/2006.14822.pdf) will be our loss function. To measure the performance we will use Accuracy and Dice Score. On this U-Net we will run the MorphNet approach with the target model size and investigate two different regularization techniques which will be discussed in the following section. Additionally, a simple [convolutional neural network](https://github.com/markusdiekmann95/Morph-U-Net/blob/main/MorphCNN.ipynb) was set up to test the feasibility on an already known task. For this purpose, the CIFAR10 Data Set **[10]** was taken and a simple network was implemented on which MorphNet is executed.

### Regularization

In the section *Network Shrinking* regularization was not specified up to now and it was just labeled as $R$. Now we will investigate two different approaches for it: *Gamma L1 Regularization* and *Group Lasso Regularization*. First we will discuss how these approaches work. Then the feasibility will be investigated and how it affects the U-Net architecture and the performance. 

#### Gamma L1 Regularization

The first regularization approach is *Gamma L1 Regularization* and this is the regularization technique that was explained and applied in the MorphNet paper **[6]**. With this regularization technique *L1 Regularization* is applied to the gammas of the *Batch Normalization* **[11]**. Therefore, we should first have a look at how *L1 Regularization* and *Batch Normalization* works. 

*L1 Regularization* **[12]** sums up all absolute values of weights and results into this formula:
\begin{aligned}\lambda * \sum |weight_i|\end{aligned} The $\lambda$ is the regularization strength and the higher lambda will be defined, the stronger the network will be regularized. This will lead to a trade-off between regularization and the performance of the network. However, L1 regularization aims to shrink inefficient features what could help reduce the widths of the network.

To apply *Batch Normalization* on a layer, it will take the values $x$ of a mini-batch $B$ as input and computes the mean $\mu_B$ and the variance $\sigma_B ^2$ of these values and it will be multiplied by a scale $\gamma$ and a bias $\beta$ will be added. This results into this formula: \begin{aligned}\hat{y}_i = \gamma *(x_i - \mu_B)/\sqrt{\sigma_B ^2 + e} + \beta\end{aligned} The scale and the shift are learnable parameters. As we apply the *L1 Regularization* on this scale, we use this scale to identify neurons that are alive and which are not alive. If a $\gamma_i$ is below the defined threshold after shrinking, the network could be set up without the neuron. To compute the costs with *Gamma L1 Regularization* is just summing up the $\gamma$-values for a layer: $\sum |\gamma_i|$. 

A simple example would be that we have four neurons with the initial *Batch Normalization* weights $[1,1,1,1]$. After applying the *Gamma L1 Regularization* on this network with the threshold $0.001$, our values for $\gamma$ are $[0.1,0.00014,0.073,0.002]$. The second value is below the threshold and will be zeroed out and this results in our shrunk model with three remaining neurons: $[1,1,1]$. 

```
def gamma_regularizer(model, threshold=0.0001):
  '''  takes  the model and threshold as input and computes the costs based on
   Gamma L1 Regularization  '''
  last_module = None
  costs = 0.
  for name, param in model.named_parameters():
    if ("conv" in name) and ("weight" in name): # Select Convolution Layer Weights
      model_size = model_size_C(param)
    if ("bn" in name) and ("weight" in name): # Select BN Weights
      if last_module is None:
        #Input
        costs += model_size * zero_out_indicator(param, threshold) * 0 #Model Input - has no gammas
        #Output
        costs += model_size * l1_reg(param) * 3 #First Batchnorm gammas in double conv block, Input channels are always alive!

        last_module = param
      else:  # In the  computation the output of the previous layer is relevant
        #Input
        costs += model_size * l1_reg(last_module) * zero_out_indicator(param) #First Batchnorm gammas in double conv block
        #Output
        costs += model_size * l1_reg(param) * zero_out_indicator(last_module,threshold) #First Batchnorm gammas in double conv block

        last_module = param #To remember the params from the previous layer
  
  return costs
  ```
First, the MorphNet Regularization was applied on the simple classification [CNN](https://github.com/markusdiekmann95/Morph-U-Net/blob/main/MorphCNN.ipynb) for CIFAR10 data with the following output widths: $[32,64,128,256,256]$. As we apply the regularization only on convolution layers, we could ignore the dense layers first. After applying regularization, we could eliminate the following proportion of weights: $[0.0313,0.1563,0.3672,0.3477,0.3672]$. So, from the first layer $3\%$ could be eliminated and from the last layer $37\%$. In Figure 3, the $\gamma$ values can be seen after sparsifying regularization and zeroing out the values below the threshold $0.0001$.
> ![](https://i.imgur.com/FLTjtgn.jpeg)
> Figure 3: Gamma Values for Classification

From this concludes our new network will be with these output widths: $[31,54,81,167,162]$. As mentioned in the paper, it could be seen clearly, that model size regularization focuses on shrinking the layers which are later in the network. The next step would be to expand the network structure again. However, we want to focus on applying the MorphNet on the U-Net and the study about applying MorphNet on classification helped to investigate if MorphNet regularization works.

After a successful test for classification tasks, we are ready to test it on U-Net for satellite images from Vienna. However, problems have arisen here: The $\gamma$-values all have identical values after applying MorphNet with the *Gamma L1 Regularization* as can be seen in Figure 4. Since the values are all equal, it would not be possible to zero out neurons and reduce the output widths.
> ![](https://radosgw.public.os.wwu.de/pad/uploads/upload_d141fd993b19d23833ab5fc002f95131.png)
> Figure 4: Gamma Values for U-Net

Thus, some [investigations](https://github.com/markusdiekmann95/Morph-U-Net/blob/main/Morph_U_Net_Inria_Vienna_Gamma_Investigation.ipynb) for the cause of this issue were conducted. First, the $\gamma$-values were investigated without applying MorphNet on the U-Net. As can be seen in Figure 5, all values are about $1$.
> ![](https://i.imgur.com/z4Gr2D4.jpg)
> Figure 5: Gamma Values for U-Net

Increasing the regularization strength also had no impact on it since all values were decreased equally. Since the problem is that the $\gamma$-values all take on the same values, it was investigated how this would be affected if the $\gamma$-values were also initialized with [random weights](https://github.com/markusdiekmann95/Morph-U-Net/blob/main/Morph_U_Net_Inria_Vienna_Gamma_Investigation.ipynb) instead of $1$, just like other weights. For this purpose, the $\gamma$-values were weighted with a *uniform distribution* $[0,1]$. How this affects the regularization can be seen in Figure 6. On the left are the values after initialization and on the right after regularization.

> ![](https://i.imgur.com/zgRJZvU.jpg)
> Figure 6: Gamma Values with random initialization

If you look at the values on the right, you can see that they are all different. But when comparing them with the values directly after initialization, you can see that the difference is the same for all of them. For example, 3 were highlighted and the $\Delta$ is $0.1315$ for all of them. Thus, this was not what we want to achieve. Further investigations were undertaken: the optimizer was changed from *ADAM* to *Stochastic Gradient Descent*. This results in the $\gamma$-values within a layer stay still the same. However, the values vary between different layers. Unfortunately, it is also not purposeful. Another consideration was that the skip connections of the U-Net are responsible for this. Therefore, these were removed, and thus MorphNet was tested on an [*Encoder-Decoder*](https://github.com/markusdiekmann95/Morph-U-Net/blob/main/Morph_Encoder_Decoder_Inria_Vienna.ipynb) for Semantic Segmentation instead of on the U-Net. Unfortunately, this was also without success. 

Concluding to *Gamma L1 Regularization*, it seems not to work on the U-Net. Several investigations were conducted, but the reason why could not be found. Therefore, an alternative regularization method must be investigated.

#### Group Lasso Regularization

As an alternative regularization approach we will investigate *Group Lasso Regularization* which was mentioned in the [github repository](https://github.com/google-research/morph-net) **[13]** as an morphnet regularization technique if no *Batch Normalization* is included in the network. But first, we will have a look at how *Group Lasso Regularization* works:

The weights of a network will be assigned into groups. For each group, the square-root of the number of weights in a group will be multiplied with the euclidean norm of all weights in this group that results in the following formula **[14]**:
\begin{aligned}\sum \sqrt{p_l} * ||\theta_l||_2\end{aligned} *Group Lasso Regularization* will be applied on the convolution layers and a single convolution layer will be considered as a group. However, the last convolution layer, which is responsible for the output, will not be affected by this regularization. This regularization will decrease the convolution weights and if a weight is below the threshold, it will be zeroed out. Based on the remaining weights in percentage, the new number of output channels could be determined for a specific convolution layer. It will be demonstrated on the following example: The threshold for zeroing out will be initiated as $0.001$ and the convolution layers of the seed network consist of these output widths: $[64,128,256,512]$. After applying MorphNet with *Group Lasso Regularization* on it, these are the remaining weights in proportion per layer: $[0.9375, 0.8672, 0.7422, 0.6641]$. These proportions will be multiplied with the initial output widths and this results in the shrunk model with the output widths $[60,111,190,340]$. The implementation is similar to the implementation of *Gamma L1 Regularization*.

```
def group_lasso_regularizer(model, threshold=0.0001):
  ''' Takes the model and threshold as input and 
  computes the costs based on Group Lasso Regularization'''
  last_module = None
  costs = 0.
  for name, param in model.named_parameters():
    if ("conv" in name) and ("weight" in name): # Select Convolution Layer Weights
      if last_module is None: #If the layer is the first convolution layer
        costs += param.size()[2]*param.size()[3] * zero_out_indicator(param.cuda()) * 0 # No regularization on input channels
          #Output
        costs += param.size()[2]*param.size()[3] * group_lasso_reg(param) * 3 # Input channels are always alive
        last_module = param 
      else: # In the  computation the output of the previous layer is relevant
          #Input
        costs += param.size()[2]*param.size()[3] * group_lasso_reg(last_module) * zero_out_indicator(param.cuda(),threshold) 
          #Output
        costs += param.size()[2]*param.size()[3] * group_lasso_reg(param) * zero_out_indicator(last_module.cuda(),threshold) 

        last_module = param #To remember the params from the previous layer
  
  return costs
  ```
The Group Lasso Regularization could be tested successfully on the U-Net for
the Inria Data Set. Thereby, MorphNet was applied [once](https://github.com/markusdiekmann95/Morph-U-Net/blob/main/Morph_U_Net_Inria_Vienna.ipynb) with low threshold $(0.0001)$ and low regularization strength ($1*10^{-10})$ and [once](https://github.com/markusdiekmann95/Morph-U-Net/blob/main/Morph_U_Net_Inria_Vienna_Strong_Reg.ipynb) with a higher threshold $(0.001)$ and stronger regularization ($1*10^{-3})$. Two iterations of MorphNet were executed on both settings.

|                | Base Model | Shrinked Model 1 | Expanded Model 1 | Shrinked Model 2 | Expanded Model 2 |
| -------------- | ---------- | -------------- | -------------- | ---------------- | ---------------- |
| Parameter      | 31.000.000 | 10.000.000     | 20.000.000     | 6.000.000        | 19.000.000       |
| Time           | 50:32      | 31:21          | 44:52          | 26:48            | 47:47            |
| Train Acc.| 0.95       | 0.96           | 0.95           | 0.97             | 0.96             |
| Train Dice     | 0.90       | 0.91           | 0.90           | 0.93             | 0.91             |
| Val Acc.   | 0.94       | 0.94           | 0.94           | 0.94             | 0.94             |
| Val Dice       | 0.87       | 0.87           | 0.87           | 0.86             | 0.87             |
> Table 1: MorphNet on U-Net with low threshold and weak regularization

The U-Net is initialized with roughly $31$ million parameters. After the first shrinking phase, the model size could be reduced to $1/3$ of the initial number of parameters. However, it keeps the same performance. This could be an indicator that the initial network is highly overparameterized and to handle this, is the task of MorphNet. For expanding stage, we set an upper border of model parameters to $20$ million and on it we initialized the width multiplier. Expanding back to the upper border does not improve the overall performance. After shrinking the network again only $1/5$ of the parameters are remaining. The performance is almost the same and expanding the network again to the upper border could not exceed the initial performance. 

|                | Base Model | Shrinked Model | Expanded Model | Shrinked Model 1 | Expanded Model 1 |
| -------------- | ---------- | -------------- | -------------- | ---------------- | ---------------- |
| Parameter      | 31.000.000 | 1.300.000      | 3.000.000      | 1.400.000        | 3.000.000        |
| Time           | 50:32      | 12:50          | 19:40          | 11:47            | 18:04            |
| Train Acc. | 0.95       | 0.95           | 0.96           | 0.96             | 0.96             |
| Train Dice     | 0.90       | 0.90           | 0.92           | 0.92             | 0.91             |
| Val Acc.   | 0.94       | 0.93           | 0.94           | 0.94             | 0.94             |
| Val Dice       | 0.87       | 0.84           | 0.86           | 0.85             | 0.86             |           |
> Table 2: MorphNet on U-Net with high threshold and strong regularization

The strong regularization with the higher threshold could decrease the model size from $31$ Million parameters to $1.3$ Million parameters. This has resulted in a slightly lower Dice Score. The upper border for this scenario is $3$ Million parameters that is roughly $10\%$ of the initial model size. Expanding to the upper border increased the performance almost back to initial performance. However, a second iteration of the MorphNet algorithm does not yield much and performance tends to stagnate.

Concluding, it can be seen that the MorphNet approach is able to handle overparameterization of the network in eliminating inefficient weights. It is possible to keep the performance of the network and in parallel, the model size can be significantly (in this case down to $10\%$ of the initial model size) reduced.

### Symmetry

An important factor which should not be neglected is the symmetry of the U-Net. The U-Net has a symmetrical architecture with skip connections from the left to the right side. Therefore, it should be examined how and to what extent this symmetry should be maintained. For this purpose, three levels were selected and investigated. The fact that the number of feature maps must double in the contracting path and halve in the expansive path was disregarded, since this is too close to the initial architecture. Then too many inefficient neurons would remain in the network. 

Level 1 is the weakest symmetry level and only requires that the number of transpose 2D convolution outputs matches the number of outputs through the skip connection. As can be seen in Figure 7, after applying MorphNet on the U-Net 48 feature maps were in the skip connection and so, the output width of the transpose 2d convolution is adjusted to 48.
> ![](https://radosgw.public.os.wwu.de/pad/uploads/upload_d0103194441fff20adbbb948c88aceaf.png)
> Figure 7: Symmetrie Level 1: Weak

Level 2 is the medium symmetry level and requires in addition to level 1, that the number of output widths within a double convolution block is equal. To avoid that important feature maps could not be lost, the maximum of both will be taken. So, on the left side the output width is for both convolution layers $max(60,48)=60$ and on the right side it is $29$. This also has an impact on the output width of the transpose 2d convolution which would be $60$. 
> ![](https://radosgw.public.os.wwu.de/pad/uploads/upload_c3beb2118d52148cb529408dbeef3ead.png)
> Figure 8: Symmetry Level 2: Medium

The strongest degree of symmetry is level 3 and it requires that all convolution layers on a line have the same number of output widths. For this again the maximum of all four layers is taken. For the example, the new output width will be $max(60,48,16,29)=60$. This retains the symmetry of the network. However, more feature maps result in higher model size. Thus, its effectiveness of it must be studied. For this purpose, the shrunk model was trained on all three symmetry levels.
|           | Level 1   | Level 2    | Level 3    |
| --------- | --------- | ---------- | ---------- |
| Parameter | 5,492,792 | 13,307,525 | 13,947,718 |
| Time      | 23:36     | 32:03      | 42:16      |
| Accuracy  | 0.96      | 0.96       | 0.96       |
| Dice      | 0.96      | 0.92       | 0.92       |
| Accuracy  | 0.94      | 0.94       | 0.94       |
| Dice      | 0.87      | 0.86       | 0.87       |
> Table 3: Symmetry Level - Performance Comparison

Based on the results of this study, it could be said that a higher symmetry level has no impact on the performance of this segmentation task. However, it could be reasonable to check that again when applying Morph-U-Net on another dataset. 

### Observation

When examining how the MorphNet affects individual layers, it can be confirmed as explained in the paper that model size optimization focuses on the later layers. As can be seen in Table 5, the last two layers have been shrunk significantly more than e.g. the first layer or the huge bottom layer. In addition, an interesting observation was made with the strong regularization. For each double convolution block, an entire convolution layer was eliminated. So it seems possible that for this task it is sufficient to replace the double convolution blocks with single convolution layers. Since in this study the goal was not to eliminate whole layers automatically, to avoid errors, a parameter with minimum layer size was implemented, which then sets the layer to $10\%$ of the initial layer size, as in this case.

|                       | Layer      | Base Model | Shrinked Model | Expanded Model | Shrinked Model 1 | Expanded Model 1 |
| --------------------- | ----------- | ---------- | -------------- | -------------- | ---------------- | ---------------- |
| Strong Regularization | 1st         | 64/64      | 57/0           | 86/9           | 77/0             | 112/9            |
| Strong Regularization |  Bottom     | 1024/1024  | 0/607          | 153/910        | 102/671          | 148/972          |
| Strong Regularization | Last        | 64/64      | 0/32           | 9/49           | 0/29             | 9/42             |
| Strong Regularization | Second Last | 128/128    | 0/60           | 19/90          | 0/0             | 18/18            |
> Table 4: Observation concerning convolution blocks

Based on these observations, a [U-Net is trained](https://github.com/markusdiekmann95/Morph-U-Net/blob/main/Morph_U_Net_1_Conv_Inria_Vienna.ipynb) which no longer consists of double convolution blocks but of single convolutions. The performance can be investigated in Table 5.

|            | Base Model | Shrinked Model A | Expanded Model A | Shrinked Model B | Expanded Model B |
| ---------- | ---------- | ---------------- | ---------------- | ---------------- | ---------------- |
| Parameter  | 15.000.000 | 3.200.000        | 5.700.000        | 750.000          | 1.700.000        |
| Time       | 34:25      | 15:32            | 21:42            | 12:50            |    21:33              |
| Train Acc. | 0.97       | 0.96             | 0.96             | 0.94             |   0.95               |
| Train Dice | 0.93       | 0.92             | 0.91             | 0.88             |       0.90           |
| Val Acc.   | 0.94       | 0.94             | 0.94             | 0.93             |       0.93           |
| Val Dice   | 0.86       | 0.86             | 0.86             | 0.84             |      0.85            |
> Table 5: Performance of MorphNet on 1-Conv-U-Net

As expected, the performance did not decline after training U-Net with single convolution layers instead of double convolution blocks. MorphNet was applied on this network once (A) with the same regularization strength ($1*10^{-3}$) and threshold ($0.001$) as from the network in Table 2 and once (B) with even stronger regularization ($1*10^{-2}$) and a higher threshold  ($0.01$). For A there is no real degradation of performance. For B the network becomes a bit worse after shrinking but the expanding stage has brought it back close to the initial performance with only $1/18$ of parameters of the U-Net with double convolution blocks.

## Conclusion

In the context of TinyML, deep learning architectures will be optimized to run on resource-constrained devices. In this experiment, the focus was on optimizing the model size on the U-Net. For this purpose, the MorphNet approach was applied. MorphNet is based on target-oriented optimization of networks in combination with sparsifying regularization and width multiplier. First of all, a problem arose and that was that the *Gamma L1 Regularization* did not work on the U-Net, because the $\gamma$-values of *Batch Normalization* within a layer are all the same, which should serve as a scale to detect inefficient neurons. Several investigations were made to find the cause. Unfortunately, the cause could not be found. As an alternative, *Group Lasso Regularization* was implemented and gave promising results. The U-Net could be freed from overparameterization and achieves a similar performance as the seed network with $10\%$ of the parameters. Thus, the model size can be greatly reduced without any real loss of performance, which may enable networks to run on resource-constrained devices.

The study demonstrates the success of applying MorphNet on the U-Net. Due to time and resource limitations, this has so far been tested on a single simple dataset. In the future, further testing on different datasets is recommended. What has been disregarded so far is that more layers could be added or removed automatically. This is likely to prove difficult due to the skip connections within the U-Net. However, it was mentioned in the GitHub repository that there is generally a way to do this. The regularization method *Logistic Sigmoid Regularization* was mentioned **[13]**. That would be an interesting step to test this on the U-Net.
Also, so far there has only been the focus on model size. Further investigations in terms of optimization of FLOPs or latency, would also be interesting research topics.

## Acknowledgement

My thanks go out to Thorben Hellweg, Moritz Seiler and Professor Dr. Fabian Gieseke from the Department of [Data Science: Machine Learning and Data Engineering](https://www.wi.uni-muenster.de/department/dasc) topic within the seminar and for supporting me by answering questions, giving me feedback and advice. In addition my thanks go out to [Mergim Mustafa](https://www.linkedin.com/in/mergim-mustafa-b0423a1a4/) due to the nice talks about U-Net.

## References

[1] Moyer, B., Why TinyML Is Such A Big Deal (2021). https://semiengineering.com/why-tinyml-is-such-a-big-deal/

[2] Luber, S., Was ist TinyML? (2022). https://www.bigdata-insider.de/was-ist-tinyml-a-1087998/

[3] Arun, An Introduction to TinyML (2020). https://towardsdatascience.com/an-introduction-to-tinyml-4617f314aa79

[4] Lin, J., Chen, W., Lin, Y., Cohn, J., Gan, C., and Han, S. (2020). “MCUNet: Tiny Deep Learning on IoT Devices,” https://arxiv.org/abs/2007.10319

[5] Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation

[6] Gordon, A., Eban, E., Nachum, O., Chen, B., Wu, H., Yang, T.-J., and Choi, E. (2018). “MorphNet: Fast Simple Resource-Constrained Structure Learning of Deep Networks,”. https://arxiv.org/abs/1711.06798

[7] Rodriquez, J., MorphNet is a Google Model to Build Faster and Smaller Neural Networks (2021). https://pub.towardsai.net/morphnet-is-a-google-model-to-build-faster-and-smaller-neural-networks-f890276da456

[8] Poon, A., MorphNet: Towards Faster and Smaller Neural Networks (2019). https://ai.googleblog.com/2019/04/morphnet-towards-faster-and-smaller.html

[9] Emmanuel Maggiori, Yuliya Tarabalka, Guillaume Charpiat and Pierre Alliez. “Can Semantic Labeling Methods Generalize to Any City? The Inria Aerial Image Labeling 
Benchmark”. IEEE International Geoscience and Remote Sensing Symposium (IGARSS). (2017).

[10] Krizhevsky, A., Nair, V., Hinton, G., The CIFAR-10 dataset https://www.cs.toronto.edu/~kriz/cifar.html

[11] Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.

[12] Nagpal, A., L1 and L2 Regularization Methods (2017).  https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c

[13] Google Research - Github Repository https://github.com/google-research/morph-net

[14] Bühlmann, P., van de Geer, S. (2011). Statistics for High-Dimensional Data, ppp. 55-76

