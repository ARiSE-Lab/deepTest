# DeepTest: Automated testing of deep-neural-network-driven autonomous cars

DeepTest is a systematic testing tool for automatically detecting erroneous behaviors of DNN-driven vehicles that can potentially lead to fatal crashes.

## Install Required Packages

Read through and run [./install.sh](./install.sh)

## Code Directories

[models/](models/)  

Reproducing Udacity self-driving car Challenge2<sup>[1]</sup> results for Rambo, Chauffeur and Epoch models.  
* [Models and weights files](https://github.com/ARiSE-Lab/deepTest#models-and-saved-weights) are required to run these scripts.
* For Rambo model, Keras backend should be changed to theano.  

[testgen/](testgen/)
* Generate synthetic images, calculate cumulative coverage and record predicted outputs
* [ncoverage.py](testgen/ncoverage.py) <sup>[2]</sup>  

[guidedsearch/](guidedsearch/)  

[metamorphictesting/](metamorphictesting/) 

## Models and [Saved Weights](https://github.com/udacity/self-driving-car/tree/master/steering-models/evaluation)    
* Rambo <sup>[3]</sup>  
* Chauffeur <sup>[4]</sup>
* Epoch <sup>[5]</sup>

 
## Datasets

* [HMB_3.bag](https://github.com/udacity/self-driving-car/blob/master/datasets/CH2/HMB_3.bag.tar.gz.torrent): Test dataset  
* [CH2_001](https://github.com/udacity/self-driving-car/tree/master/datasets/CH2): Final Test Data for challenge2
  * [CH2_001 labels](https://github.com/udacity/self-driving-car/blob/master/challenges/challenge-2/CH2_final_evaluation.csv)
## Detected erroneous behaviors
https://deeplearningtest.github.io/deepTest/

## Citation
If you find DeepTest useful for your research, please cite the following [paper](https://arxiv.org/pdf/1708.08559.pdf):

```
@article{tian2017deeptest,
  title={DeepTest: Automated testing of deep-neural-network-driven autonomous cars},
  author={Tian, Yuchi and Pei, Kexin and Jana, Suman and Ray, Baishakhi},
  journal={arXiv preprint arXiv:1708.08559},
  year={2017}
}

```
## References

1.  **Udacity self driving car challenge 2**. <br /> 
https://github.com/udacity/self-driving-car/tree/master/challenges/challenge-2. (2016).
2. Pei, Kexin, et al. **Deepxplore: Automated whitebox testing of deep learning systems.** <br />
3.  **Rambo model**. <br />
https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/rambo. (2016).
4.  **Chauffeur model**. <br />
https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/chauffeur. (2016).
5.  **Epoch model**. <br />
https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/cg23. (2016).

