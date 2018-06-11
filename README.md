# DeepTest: Automated testing of deep-neural-network-driven autonomous cars

DeepTest is a systematic testing tool for automatically detecting erroneous behaviors of DNN-driven vehicles that can potentially lead to fatal crashes.

## Install Required Packages

Read through and run [./install.sh](./install.sh)

## Code Directories

[models/](models/)

* Epoch model  
* Chauffeur model  
* Rambo model  

[testgen/](testgen/)

* Generate synthetic images  
* Calculate cumulative coverage and record predicted outputs

[guidedsearch/](guidedsearch/)  

[metamorphictesting/](metamorphictesting/) 


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

1.  **Rambo model**. <br />
https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/rambo. (2016).
2.  **Chauffeur model**. <br />
https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/chauffeur. (2016).
3.  **Epoch model**. <br />
https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/cg23. (2016).
