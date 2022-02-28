# Morph-U-Net
Application of the MorphNet Approach on the U-Net

This is an own PyTorch-based implementation of the MorphNet Approach to apply [MorphNet](https://arxiv.org/abs/1711.06798) on the U-Net.

Notebooks:
| Notebook                                                 | Description                                                                                  |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| MorphCNN.ipynb                                           | Applied MorphNet on a basic CNN                                                              |
| Morph\_Encoder\_Decoder\_Inria\_Vienna.ipynb             | Applied MorphNet on a U-Net without Skip Connections                                         |
| Morph\_U\_Net\_1\_Conv\_Inria\_Vienna.ipynb              | Applied MorphNet on U-Net with single convolution layers instead of double convolution blogs |
| Morph\_U\_Net\_Inria\_Vienna.ipynb                       | Main Notebook                                                                                |
| Morph\_U\_Net\_Inria\_Vienna\_Gamma\_Investigation.ipynb | Further investigation about the gamma issue                                                  |
| Morph\_U\_Net\_Inria\_Vienna\_Strong\_Reg.ipynb          | Same as Main Notebook, just applied stronger regularization                                  |
| process\_images.ipynb                                    | Prepared the images for U-Net                                                                |

This project is in context of the seminar RECENT TRENDS IN DEEP LEARNING WS 2021/22 at the chair [Data Science: Machine Learning and Data Engineering](https://www.wi.uni-muenster.de/department/dasc) at WWU Münster.
