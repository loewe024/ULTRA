# ULTRA tracking teacher

The checkpoint at `ultra/weights/teacher_ultra_inference.pth` is used for tracking inference and student distillation. It takes 4052-dimensional observations and produces 29-dimensional actions.

## Training data and usage terms

After the original OMOMO work, this checkpoint was further trained using OMOMO, AMASS, and BONES-SEED motions. The [AMASS license](https://amass.is.tue.mpg.de/license.html) restricts use of its data and models trained from it to non-commercial purposes and prohibits sharing the dataset. This weight is provided for non-commercial research and education; the top-level Apache-2.0 code license does not replace the underlying data terms.

The [BONES-SEED license](https://bones.studio/info/seed-license) defines trained models as Results, imposes eligibility and use conditions, prohibits redistribution of raw data, and requires attribution when distributing a model. Training data includes [Motion Data by Bones Studio](https://bones.studio/). Use of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement. The checkpoint is distributed by the repository maintainers; downstream users remain responsible for the applicable dataset terms.

No source dataset is included with this weight. Check the dataset terms before using or redistributing the model.
