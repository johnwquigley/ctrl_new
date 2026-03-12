# CUrrent state:

neural_train_pipeline.ipynb contains the emulator/controller training pipeline, it import modules in ctrl/neural/*, which contains data creation, model definition, training loop, etc.

truck_ctrl.ipynb is direct optimal control, in the last cell it calls ctrl/truck_data_gen.py for faster data generation to do an evaluation of ctrl signals

eval.py loads a checkpoint and evaluate it on different difficulties.

`python eval.py --inference-mode me --checkpoint  ctrl/pth/neural/controller_epoch_001.pth`

for the previous student code, as he use a different input for the controller, to evaluate, do

`python eval.py --inference-mode student`

## TODO:

1. cleanup and condense into one notebook(john)
2. After changing configs (v = -0.1, and increased control steps), some conclusion we derived in optimal control no longer holds, e.g. head supervision necessary? Adding cost terms makes the optimization applys more conservative actions to reduce action cost, instead of achieving the final 1m distance, need a revision
3. After changing configs (v = -0.1, and increased control steps), plots need revision as there are much more steps, also need to create suitable plots for neural_train_pipelines
4. need to add step 2(forgot what it is called), supervised learning, could it replace the curriculum learning, directly work on the last curriculum(currently there are two curriculums)
5. for emulator/controller diagnose the failure issue and decide whether to add the jackknife loss term
