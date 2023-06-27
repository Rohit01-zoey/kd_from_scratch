This repository contains my implementation of the knowledge distillation in machine learning

# The setting of the problem
 We look at the fully supervised setting ie the student and the teacher have labelled data(the same data).

 We distinguish the training process into two modes:
 1. The teacher is trained fully(upto reasonable loss) and then the student is trained using the teacher as a guide
 2. The teacher and the student are trained simultaneously. Here the teacher is seen as a guide througout the training of the student as though guiding is throught the training space

 # Aim
 1. Getting familiar with coding from scratch using pytorch(first time)
 2. Getting used to using cuda to accelerate model training in pytorch