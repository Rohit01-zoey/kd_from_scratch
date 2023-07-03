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

Project Name: [Enter your project name here]

Author: [Your name]

Date: [Current date]

1. Problem Statement:
   - Clearly define the problem you are trying to solve with machine learning.
   - Describe the input data, expected output, and the type of machine learning task.

2. Data:
   - Identify the data sources and gather the required dataset.
   - Describe the features, their types, and any preprocessing steps needed.
   - Specify how the data will be split into training and testing sets.

3. Modules:
   - List the main modules you plan to have in your code.
   - Briefly describe the purpose and responsibilities of each module.

4. Module 1: [Module Name]
   - Description: [Provide a brief description of the module's purpose]
   - Functions/Classes:
     - [Function/Class 1]: [Description]
     - [Function/Class 2]: [Description]
     - ...

5. Module 2: [Module Name]
   - Description: [Provide a brief description of the module's purpose]
   - Functions/Classes:
     - [Function/Class 1]: [Description]
     - [Function/Class 2]: [Description]
     - ...

6. Dependencies:
   - Identify the dependencies between different modules.
   - Specify any external libraries or packages required for your project.

7. Main Script:
   - Describe the structure and flow of your main script.
   - Outline the steps involved in executing your machine learning pipeline.

8. Testing and Evaluation:
   - Define how you plan to test and evaluate your model's performance.
   - Specify the evaluation metrics and techniques you will use.

9. Documentation:
   - Describe how you will document your code.
   - Specify any guidelines or standards you will follow for code comments and documentation.

10. Next Steps:
    - Outline the next steps you plan to take for your project.
    - Identify any potential challenges or areas where you may need further research.

11. Conclusion:
    - Summarize your overall plan and the goals you aim to achieve.


 # References
 1. https://editor.analyticsvidhya.com/uploads/30818Knowledge%20Distillation%20Flow%20Chart%201.2.jpg