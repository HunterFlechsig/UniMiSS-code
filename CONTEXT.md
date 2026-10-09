# Chest image classification

Labeled chest radiographs and chest CT volumes. Each collection has its own classification. CXR-COVID-19, VinDr-CXR, and RICORD remain part of this language.

## Language

**Radiograph**:
One frontal chest X-ray image.
_Avoid_: study, scan, case, sample

**CXR-COVID-19**:
The chest X-ray collection in which each radiograph is exactly one of COVID, normal, or pneumonia.
_Avoid_: COVID dataset, the 2D dataset

**CXR-COVID-19 classification**:
Assigning a CXR-COVID-19 radiograph to exactly one of COVID, normal, or pneumonia.
_Avoid_: 2d cls, the original plan

**VinDr-CXR**:
The VinBigData collection of 18,000 adult PA chest radiographs. Each of the 15,000 training radiographs has three independent radiologist marks. Each of the 3,000 test radiographs has one consensus of five radiologists.
_Avoid_: VinDr, VinDr-Mammo, VinDr-PCXR, VinDr-SpineXR

**Local finding**:
One of the 22 abnormalities a radiologist localizes with a box on a VinDr-CXR radiograph, from Aortic enlargement through Other lesion.
_Avoid_: class, disease, diagnosis

**Global diagnosis**:
One of the six image-level impressions on a VinDr-CXR radiograph: Lung tumor, Pneumonia, Tuberculosis, Other diseases, COPD, or No finding. VinDr-CXR Pneumonia is not the CXR-COVID-19 pneumonia category.
_Avoid_: finding, class

**No finding**:
The global diagnosis in which the radiologist recorded no local finding. A mark cannot both draw a local finding and select No finding. It is also the sixth bit of the PE-global label.
_Avoid_: normal, negative

**Pleural effusion**:
A local finding that VinDr-CXR classification scores as an image-level bit.
_Avoid_: PE, pulmonary embolism

**PE-global label**:
The six-bit label of VinDr-CXR classification, in order: Pleural effusion, Lung tumor, Pneumonia, Tuberculosis, Other diseases, No finding. COPD is not a bit. Several bits may be on for one radiograph. The prepared training list and test list are this label; the bits are not recomputed from radiologist marks.
_Avoid_: global diagnosis, the official six

**Fit list**:
The radiographs left in the training list after the validation list is removed. Weights are updated on these radiographs.
_Avoid_: train set

**Validation list**:
Ten percent of the training list, held out by radiograph and stratified so each bit of the PE-global label keeps a similar positive rate. The checkpoint is chosen here. Each bit has at least one positive and one negative radiograph on this list.
_Avoid_: test

**Test list**:
The prepared test radiographs, scored once after the checkpoint is chosen.
_Avoid_: validation

**Headline score**:
The unweighted mean of per-label ROC-AUCs on a collection's own labels. CXR-COVID-19 classification's AUC is this mean over its three categories. The PE-global headline score is this mean over its six bits.
_Avoid_: micro AUC, accuracy

**Reported scores**:
The headline score, the six per-bit ROC-AUCs, and the macro average precision. Macro average precision is the unweighted mean of the six per-bit average precisions. These scores are computed on the validation list and, once, on the test list. The checkpoint uses the headline score alone.
_Avoid_: accuracy, subset accuracy

**Checkpoint**:
The weights from the epoch with the highest headline score on the validation list.
_Avoid_: last epoch

**Radiologist mark**:
The labels one radiologist assigned to one training radiograph. The test consensus is a single agreed label, not a radiologist mark.
_Avoid_: ground truth, annotation file

**VinDr-CXR classification**:
Image-level classification of a VinDr-CXR radiograph onto the PE-global label. Scored on its own, beside CXR-COVID-19 classification. A box is not the label of this classification.
_Avoid_: detection, localization, the CXR-COVID-19 run

**RICORD volume**:
One preprocessed CT volume from MIDRC-RICORD-1A or MIDRC-RICORD-1B.
_Avoid_: radiograph, VinDr-CXR

**RICORD classification**:
Assigning a RICORD volume to one of two categories. In the shipped lists, a 1A volume is category 1 and a 1B volume is category 0.
_Avoid_: VinDr-CXR classification, CXR-COVID-19 classification

**Smoke run**:
One epoch of VinDr-CXR classification, including the test-list scoring that follows that epoch, and one epoch of RICORD classification. Started by hand on a Sol compute node.
_Avoid_: Slurm job

**CheXpert**:
The Stanford collection of chest radiographs labeled for 14 observations. Each observation is positive, negative, uncertain, or unmentioned.
_Avoid_: CheXpert-14, MIMIC-CXR

**CT-RATE**:
The collection of non-contrast chest CT volumes with report-extracted labels for 18 abnormalities.
_Avoid_: RICORD, RAD-ChestCT

**CT-RATE volume**:
One chest CT volume from CT-RATE.
_Avoid_: radiograph, RICORD volume

**Labeled student-teacher**:
Labeled training of a student and a teacher. The student encoder, consistency head, and current task head take the label loss. A consistency loss matches the two consistency heads. The teacher encoder and consistency head are an exponential moving average of the student, updated at the end of a collection epoch, and they get no gradient. This is not UniMiSS+ pretraining, and it is not the choice of which pretrained weights are loaded.
_Avoid_: DINO, teacher checkpoint

**Shared finding**:
A finding that at least two collections name, even when the rest of their labels differ.
_Avoid_: the same label list, overlapping representation

**PadChest**:
The San Juan Hospital collection of chest radiographs, labeled with its own radiographic findings. Those findings are not rewritten into the CheXpert observations.
_Avoid_: BIMCV, MIMIC-CXR

**RAD-ChestCT**:
The Duke collection of chest CT volumes. The public release contains 3,630 volumes, with a publisher validation list and a publisher test list. The checkpoint is chosen on the validation list. The test list is scored once. This classification reads abnormality labels as image-level presence.
_Avoid_: CT-RATE, the full 36,316

**Primary pair**:
CheXpert and CT-RATE.
_Avoid_: the four collections, the secondary pair

**Secondary pair**:
PadChest and RAD-ChestCT.
_Avoid_: the primary pair, MIMIC-CXR

**Task head**:
The classifier for one collection, trained on that collection's own labels.
_Avoid_: a shared head, the merged label list

**Consistency head**:
The layer that maps the student encoder output and the teacher encoder output into one space for the consistency loss.
_Avoid_: projector, task head

**Individual run**:
A labeled student-teacher with its own weights, trained on one collection and scored on that collection. It does not share weights with a cyclic run.
_Avoid_: a shared model, continued training

**Cyclic run**:
A labeled student-teacher with its own weights. It visits the collections in the run, in cycle order, and is scored on every one of those collections. Its weights do not come from an individual run.
_Avoid_: continued training, the individual run

**Pretrained start**:
The option to load the UniMiSS+ teacher encoder into both the student and the teacher, or to load it into neither. Task heads and consistency heads start untrained either way.
_Avoid_: loading only the student, the UniMiSS+ student encoder

**Uncertain observation**:
A CheXpert mark that names an observation without calling it present or absent. It is not a training target, and it is not counted in that observation's ROC-AUC.
_Avoid_: negative, unmentioned, blank

**Unmentioned observation**:
A CheXpert observation the labeler did not mention. It counts as negative.
_Avoid_: uncertain observation

**CheXpert validation list**:
The 200 studies, 234 radiographs, labeled by a majority of three radiologists. The checkpoint is chosen here.
_Avoid_: the test list, train.csv

**CheXpert test list**:
The 500 studies, 668 radiographs, labeled by a majority of five radiologists across 14 observations. Scored once after the checkpoint is chosen.
_Avoid_: the validation list, the five competition observations

**CT-RATE validation list**:
The publisher's 1,304 patients, held out by patient. There is no separate public test list. The checkpoint is chosen here, and this list is the reported score.
_Avoid_: the training patients

**PadChest validation list**:
Ten percent of PadChest patients, held out by patient. The checkpoint is chosen here, and this list is the reported score.
_Avoid_: the hand-labeled reports

**Metrics record**:
The row written on the validation list after a collection has been trained. It holds the training loss, the validation loss, the headline score, each label's ROC-AUC, the macro average precision, and each label's average precision. A cyclic run appends one row per collection visit. An individual run appends one row per epoch.
_Avoid_: the checkpoint

**Round**:
One pass of a cyclic run through the cycle order. Each collection in that pass is trained for one epoch.
_Avoid_: a visit of several epochs

**Resume**:
Continuing a run that stopped, from the last finished epoch, with that run's weights. The metrics record stays in place.
_Avoid_: pretrained start, a new run

**Concurrent run**:
A labeled student-teacher whose mini-batch draws from every collection in the run, with each image scored by its own task head.
_Avoid_: the cyclic run

**Cycle order**:
The sequence in which a cyclic run visits collections. It is an argument of that experiment.
_Avoid_: a fixed order

**Experiment**:
A named run with its own weights, its own run list, and its own arguments. A new experiment does not load another experiment's checkpoint. Resume continues the experiment with that name.
_Avoid_: the primary-pair checkpoint, a shared model

**Run list**:
The collections in one run, chosen for that run. It may be the primary pair, the secondary pair, both, or a longer list.
_Avoid_: the four collections, a fixed set

**Added collection**:
A chest radiograph collection or a chest CT collection beyond the primary and secondary pairs. It keeps its own labels, its own task head, and its own validation list.
_Avoid_: a merged label list, a shared head
