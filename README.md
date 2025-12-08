# rcnn-aux-variables

ref:
- Thorat, S., Aldegheri, G., & Kietzmann, T. C. (2021, November 15). Category-orthogonal object features guide information processing in recurrent neural networks trained for object categorization. arXiv.org. https://arxiv.org/abs/2111.07898


replication results with slight modifications / variations to the original implementation and analysis:

### Input modulation through recurrent flow from later convolutional layers to input.
[![Input Modulation](./plots/input_modulations.png)](./plots/input_modulations.png)   
### Auxiliary variable decoding across layers and timesteps:
[![Auxiliary Variable Decoding](./plots/decoding_all_variables.png)](./plots/decoding_all_variables.png)
### perturbation analysis of auxiliary variable information in the recurrent information flow:
#### example of location perturbations applied to input images:
[![Perturbation input example](./plots/perturbation_examples.png)](./plots/perturbation_examples.png) 
#### accuracy after location-perturbation vs control(randome) perturbation of matched magnitude (displacment of equal norms):  
[![Perturbation accuracy results](./plots/perurbation_analysis_accuracy.png)](./plots/perurbation_analysis_accuracy.png)
#### functional importance of auxiliary variable information in the recurrent information flow:
[![functional importance results](./plots/functional_importance_recurrent_flow.png)](./plots/functional_importance_recurrent_flow.png)
