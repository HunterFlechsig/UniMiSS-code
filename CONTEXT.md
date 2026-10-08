# Chest radiograph classification

Two labeled chest X-ray collections, each with its own classification. RICORD classification is a separate CT classification and is not one of these collections.

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
The unweighted mean of per-bit ROC-AUCs. CXR-COVID-19 classification's AUC is this same mean over its three categories.
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
