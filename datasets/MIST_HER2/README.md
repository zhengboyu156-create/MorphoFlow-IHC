# MIST HER2 paired images

Unmodified MIST HER2 JPEGs and split from MAPS-GAN data archive commit `625bc17a502e611a494e0ab1e5b570efa3a347d9`, `data/MIST_HER2_TrainValAB`.

| Split | Images | Modality |
|---|---:|---|
| trainA | 4,642 | H&E |
| trainB | 4,642 | HER2 IHC |
| valA | 1,000 | H&E |
| valB | 1,000 | HER2 IHC |

Files are grouped by original filename prefix and paired by filename. `manifest.csv` lists original SHA-256 and bytes. Patient independence has not been established.

Images are Git LFS objects. Run `git lfs install` before cloning, or `git lfs pull` afterward. This archive contains HER2 only.

- Source archive: https://github.com/zhengboyu156-create/MAPS-GAN-CGA-BES-CDM/tree/625bc17a502e611a494e0ab1e5b570efa3a347d9/data/MIST_HER2_TrainValAB
- Dataset authors and citation: https://github.com/lifangda01/AdaptiveSupervisedPatchNCE#downloading-the-mist-dataset

Original dataset terms and citation requirements apply separately from this repository's code license. The source archive did not include an explicit dataset license; consult the original data publisher before reuse.
