# Introduction
Algorithms for stenosis detection in [ARCADE Challenge](https://arcade.grand-challenge.org/) held at MICCAI 2023. We are ranked 3rd!

Please refer to [MICCAI-ARCADE](https://github.com/NMHeartAI/MICCAI_ARCADE.git) for the segmentation detection task.


## Instructions
- [Installation instructions]

      git clone https://github.com/HuiLin0220/StenUNet.git
      cd StenUNet
      pip install  -r ./requirements.txt

  should keep .gitattributes, since the model file is a large file

  model weights: ./model_final.pth
  
  model structure file: ./model_folder/plans.json
  
  data configuration file: ./model_folder/plans.json

  model input: XX.png

## Dataset
<div align = center>
<img src="/illustration/examples.svg" width="375" height="250">
</div>

## Methods
<div align = center>
<img width="600" height="75" src="/illustration/pipeline.svg">
</div>

<div align = center>
<img width="550" height="200" src="/illustration/StenUNet.svg">
</div>

## Example Results
<div align = center>
<img width="252" height="180" src="/illustration/XX.png">
</div>

## References and Acknowledgements:

    [1] Our paper (waiting to be online)
    
    


## Contact Us
Feel free to contact us: huilin2023@u.northwestern.edu; tom.liu@northwestern.edu

## To-do List
