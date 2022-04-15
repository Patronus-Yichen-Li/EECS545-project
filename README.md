# EECS545-project
Introduce "link prediction" and other graph analysis techniques to stock prediction.

# Files
global_func.py: Implementation of global functions, like similarity calculation  
learn.py: (currently lies main function) For model training and optimization  
predict.py: For testing future trend  
model.py: Implementation of OptHIST model class  
model_org.py: The copy of original HIST model, for details, please refer to https://github.com/wentao-xu/hist  

# Versions
03/16: Version 1.0: implemented the model structure, received the same output format

04/14: Version 2.0: updated model.py to the format of super class nn.Module in order to do further training and testing. Rearranged the concepts functions. Implemented and tunned the training/test epoch processes, the program can run smoothly
