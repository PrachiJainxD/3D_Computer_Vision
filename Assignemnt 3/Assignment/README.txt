======================================================================================================================================
                      Intelligent Visual Computing [Spring 2023] Assignment 3: Point descriptors and Alignment
                                                  Prachi Jain (SPIRE ID: 32759600)
======================================================================================================================================

TASK D1]. Train the architecture with train_corrmask =0
===============================================================================================================
python train.py
=> Current model is the best according to validation accuracy
train_acc: 0.62326399.
val_acc: 0.57470727.
test_acc: 0.51953936.
Attached current model: model_best.pth.tar


TASK D1]. Report the correspondence (matching) accuracy at distance thresholds 0.01, 0.02 , and 0.04 
(see distance_threshold input parameter) at the best epoch (epoch with highest hold-out validation accuracy).
===============================================================================================================

For train_corrmask = 0 and distance_threshold = 0.01:
----------------------------------------------------------------------------------------------------------------
python train.py -e --distance_threshold=0.01
test_acc: 0.51985347.

For train_corrmask = 0 and distance_threshold = 0.02:
----------------------------------------------------------------------------------------------------------------
python train.py -e --distance_threshold=0.02
test_acc: 0.90177798.

For train_corrmask = 0 and distance_threshold = 0.04:
----------------------------------------------------------------------------------------------------------------
python train.py -e --distance_threshold=0.04
test_acc: 0.99840254.


TASK D2]. Train the architecture with train_corrmask = 1 
===============================================================================================================
python train.py --train_corrmask
Average fitted rotation:  [ 1.37298611 -3.31464882 -1.54167509]
Average fitted rotation:  [ 3.24140646 -6.02469745 -1.72539718]
=> Current model is the best according to validation accuracy
train_acc: 0.75032046.
val_acc: 0.74202937.
test_acc: 0.73701799.
Attached current model: model_corr_best.pth.tar

TASK D2]. Report the correspondence mask accuracy (one value) at the best epoch.
----------------------------------------------------------------------------------------------------------------
python train.py --train_corrmask -e
test_acc: 0.74143285.


TASK E]. Results for 3D rotation angles for the test set at best epoch:
===============================================================================================================
python train.py --train_corrmask -e
Average fitted rotation:  [ 2.60679794 -4.55829855 -2.67051149]
