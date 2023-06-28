This repository contains my implementation of the knowledge distillation in machine learning

# The Problem setting
 We look at the fully supervised setting ie the student and the teacher have labelled data(the same data).

The training process can be classified into two modes : 
1. 'train' :The teacher is trained till SOTA(state of the art). This teacher is then used as a soft-labeller i.e. it labels the data with soft lables (its output) and then, the student is trained using this 'new' enhanced dataset using the updated loss function
2. 'guide' : Here the teacher is seen as a guide. So after every few epochs the teacher labels the data at the student side and then the student uses this new dataset to train for a few epochs and this is repeated.


![Knowledge distillation setup using the teacher and student models](/home/hp/kd_from_scratch/kd_from_scratch/asset)
 # Aim
 1. Getting familiar with coding from scratch using pytorch(first time)
 2. Getting used to using cuda to accelerate model training in pytorch

 # References
 1. https://editor.analyticsvidhya.com/uploads/30818Knowledge%20Distillation%20Flow%20Chart%201.2.jpg