# Reference audit and verified reference list

Every reference below was checked in October 2026 against an index record (publisher,
PubMed/PMC, DBLP or arXiv) for authors, title, venue, year and identifier. The **claim**
column is the specific statement the reference supports in the revised manuscript;
nothing is cited for more than that (reviewer major 14).

## 1. Audit of the original manuscript's citations

| # | Original citation | Used for | Verdict | Action |
|---|---|---|---|---|
| 1 | Woo et al., JAMA Netw Open 2022 | 30-day ICH mortality 40-50 % | Real paper, but on **spontaneous intracerebral** haemorrhage (a stroke subtype). Does not describe a traumatic cohort. | Removed; replaced by TBI epidemiology (Maas 2022, Dewan 2019, GBD 2019). |
| 2 | Xu et al., BMC Public Health 2024 | 3.4 M cases, 3.3 M deaths, 79.5 M DALYs | Intracerebral haemorrhage burden (GBD 2021); wrong population (reviewer 11). | Removed. |
| 3 | Song et al., Eur J Neurol 2025 | same | Same problem. | Removed. |
| 4 | Soni et al., CMPB Update 2022 | "CNNs on a similar issue, 93 % accuracy" | COVID-19 chest CT classification; not related to ICH segmentation. | Removed. |
| 5 | Juluru et al., Radiol AI 2021 | "AI positively impacted ER workflow"; "their model performed with an accuracy of 93 %" | **Misrepresented.** The paper describes a deployment framework piloted on *lymphoscintigraphy* (1,748 exams); it reports no ICH model and no 93 % accuracy. | Removed. Workflow claims now cite Arbabshirani 2018 and O'Neill 2021, which studied ICH triage directly. |
| 6 | Srikrishna et al., Front Comput Neurosci 2022 | "3D U-Nets not always superior on CT" | Real (title genuinely reads "...in One-Dimensional Brain CT"); about **brain tissue classification**, not haemorrhage. | Kept, with the claim narrowed to what it shows. |
| 7 | Ronneberger et al., MICCAI 2015 | U-Net | Correct. | Kept. |
| 8 | Hssayeni, PhysioNet 2020 | dataset | Correct (v1.3.1 DOI 10.13026/4nae-zg36). | Kept, plus the dataset paper (Hssayeni et al., Data 2020). |
| 9 | NiBabel documentation | NIfTI loading | Software documentation. | Kept as a software reference. |
| 10 | DenOtter & Schubert, StatPearls | HU definition | Not peer-reviewed; definitional only. | Kept for the definition only. |
| 11 | "C. Frey, J. M. Hazenfield. Essentials of head trauma imaging. Semin Ultrasound CT MRI 39, 2018, doi 10.1053/j.sult.2018.01.004" | haemorrhage subtypes | **Could not be verified.** No article with this title/DOI was found in the indexed 2018 volume. | Removed. Subtype descriptions now cite verified sources (Monteiro 2020; Flanders 2020). **Please check your original source before resubmission.** |
| 12 | tf.keras.metrics.BinaryIoU documentation | the metric | Correct as software documentation; it confirms the metric averages lesion and background IoU by default. | Kept, now to explain why the old 0.85 was inflated. |

## 2. Verified reference list (revised manuscript)

Numbering is assigned in order of first citation by `ich/manuscript.py`.

| Key | Reference | Identifier | Claim supported |
|---|---|---|---|
| maas2022 | Maas AIR, Menon DK, Manley GT, et al. Traumatic brain injury: progress and challenges in prevention, clinical care, and research. Lancet Neurol. 2022;21(11):1004-1060. | doi:10.1016/S1474-4422(22)00309-X | TBI is a leading cause of injury-related death and disability worldwide |
| dewan2019 | Dewan MC, Rattani A, Gupta S, et al. Estimating the global incidence of traumatic brain injury. J Neurosurg. 2019;130(4):1080-1097. | doi:10.3171/2017.10.JNS17352 | ~69 million TBIs per year |
| gbd2019 | GBD 2016 Traumatic Brain Injury and Spinal Cord Injury Collaborators (James SL, et al.). Global, regional, and national burden of traumatic brain injury and spinal cord injury, 1990-2016. Lancet Neurol. 2019;18(1):56-87. | doi:10.1016/S1474-4422(18)30415-0 | global TBI burden estimates |
| hssayeni2020 | Hssayeni MD, Croock MS, Salman AD, Al-khafaji HF, Yahya ZA, Ghoraani B. Intracranial hemorrhage segmentation using a deep convolutional model. Data. 2020;5(1):14. | doi:10.3390/data5010014 | the dataset's origin (TBI patients), annotation, and its own baseline (Dice 0.31, 5-fold CV) |
| physionet | Hssayeni M. Computed tomography images for intracranial hemorrhage detection and segmentation (version 1.3.1). PhysioNet. 2020. | doi:10.13026/4nae-zg36 | data source |
| goldberger2000 | Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation. 2000;101(23):e215-e220. | doi:10.1161/01.CIR.101.23.e215 | PhysioNet (required citation) |
| wu2023bhsd | Wu B, Xie Y, Zhang Z, Ge J, Yaxley K, Bahadir S, Wu Q, Liu Y, To M-S. BHSD: a 3D multi-class brain hemorrhage segmentation dataset. MLMI 2023, LNCS 14348:147-156. | doi:10.1007/978-3-031-45673-2_15 | BHSD: 192 voxel-labelled volumes, five subtypes; low EDH performance due to rarity |
| li2023instance | Li X, Luo G, Wang K, et al. The state-of-the-art 3D anisotropic intracranial hemorrhage segmentation on non-contrast head CT: the INSTANCE challenge. arXiv:2301.03281 (2023). **Preprint.** | arXiv:2301.03281 | INSTANCE 2022: 100 training cases, 70-case test, top Dice 0.6925, ranking on DSC/HD/RVD/NSD |
| chilamkurthy2018 | Chilamkurthy S, Ghosh R, Tanamala S, et al. Deep learning algorithms for detection of critical findings in head CT scans: a retrospective study. Lancet. 2018;392(10162):2388-2396. | doi:10.1016/S0140-6736(18)31645-3 | large-scale ICH detection on head CT (CQ500) |
| kuo2019 | Kuo W, Häne C, Mukherjee P, Malik J, Yuh EL. Expert-level detection of acute intracranial hemorrhage on head computed tomography using deep learning. PNAS. 2019;116(45):22737-22745. | doi:10.1073/pnas.1908021116 | segmentation-based ICH detection, AUC 0.991 |
| flanders2020 | Flanders AE, Prevedello LM, Shih G, et al. Construction of a machine learning dataset through collaboration: the RSNA 2019 brain CT hemorrhage challenge. Radiol Artif Intell. 2020;2(3):e190211. | doi:10.1148/ryai.2020190211 | RSNA 2019 dataset (>25,000 exams, subtype labels) |
| wang2021 | Wang X, Shen T, et al. A deep learning algorithm for automatic detection and classification of acute intracranial hemorrhages in head CT scans. NeuroImage Clin. 2021;32:102785. | doi:10.1016/j.nicl.2021.102785 | RSNA 2019 winning approach (2D CNN + sequence models) |
| arbabshirani2018 | Arbabshirani MR, Fornwalt BK, Mongelluzzo GJ, et al. Advanced machine learning in action: identification of intracranial hemorrhage on computed tomography scans of the head with clinical workflow integration. npj Digit Med. 2018;1:9. | doi:10.1038/s41746-017-0015-z | prospective worklist reprioritisation cut time to diagnosis 512 -> 19 min |
| oneill2021 | O'Neill TJ, et al. Active reprioritization of the reading worklist using artificial intelligence has a beneficial effect on the turnaround time for interpretation of head CT with intracranial hemorrhage. Radiol Artif Intell. 2021;3(2):e200024. | doi:10.1148/ryai.2020200024 | reprioritisation, not notification, reduced wait time |
| ginat2020 | Ginat DT. Analysis of head CT scans flagged by deep learning software for acute intracranial hemorrhage. Neuroradiology. 2020;62(3):335-340. | doi:10.1007/s00234-019-02330-w | deployed ICH software accuracy varies by clinical setting |
| monteiro2020 | Monteiro M, Newcombe VFJ, Mathieu F, et al. Multiclass semantic segmentation and quantification of traumatic brain injury lesions on head CT using deep learning: an algorithm development and multicentre validation study. Lancet Digit Health. 2020;2(6):e314-e322. | doi:10.1016/S2589-7500(20)30085-6 | multiclass TBI lesion segmentation, multicentre validation |
| inkeaw2022 | Inkeaw P, et al. Automatic hemorrhage segmentation on head CT scan for traumatic brain injury using 3D deep learning model. Comput Biol Med. 2022;146:105530. | doi:10.1016/j.compbiomed.2022.105530 | TBI haemorrhage segmentation with subdural + bone windows; median Dice > 0.37 per subtype (full author list to be copied from the publisher page) |
| spahr2023 | Spahr A, Ståhle J, Wang C, Kaijser M. Label-efficient deep semantic segmentation of intracranial hemorrhages in CT-scans. Front Neuroimaging. 2023;2:1157565. | doi:10.3389/fnimg.2023.1157565 | transfer learning helps ICH segmentation; smaller labelled sets benefit more |
| rasoulian2023 | Rasoulian A, Salari S, Xiao Y. Weakly supervised intracranial hemorrhage segmentation using head-wise gradient-infused self-attention maps from a Swin transformer in categorical learning. Mach Learn Biomed Imaging (MELBA). 2023;2:338-360. | doi:10.59275/j.melba.2023-553a | weakly supervised Dice 0.44, comparable to supervised U-Net/Swin-UNETR |
| zheng2025 | Zheng C, Xie G, Wang H, et al. HemaContour: explicit parametric contour learning for robust ICH segmentation on non-contrast CT. npj Digit Med. 2025;8:754. | doi:10.1038/s41746-025-02124-w | trained on INSTANCE; external per-patient Dice 0.843 on PhysioNet CT-ICH |
| ronneberger2015 | Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. MICCAI 2015, LNCS 9351:234-241. | doi:10.1007/978-3-319-24574-4_28 | U-Net |
| isensee2021 | Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods. 2021;18(2):203-211. | doi:10.1038/s41592-020-01008-z | self-configuring preprocessing/training; CT normalisation |
| isensee2024 | Isensee F, Wald T, Ulrich C, Baumgartner M, Roy S, Maier-Hein K, Jäger PF. nnU-Net revisited: a call for rigorous validation in 3D medical image segmentation. MICCAI 2024, LNCS 15009:488-498. | arXiv:2404.09556 | CNN U-Nets remain state of the art under rigorous validation; inadequate baselines inflate claims |
| he2016 | He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. CVPR 2016:770-778. | doi:10.1109/CVPR.2016.90 | ResNet encoder |
| srikrishna2022 | Srikrishna M, Heckemann RA, Pereira JB, et al. Comparison of two-dimensional- and three-dimensional-based U-Net architectures for brain tissue classification in one-dimensional brain CT. Front Comput Neurosci. 2022;15:785244. | doi:10.3389/fncom.2021.785244 | 3D is not uniformly better than 2D on brain CT (tissue classification) |
| maierhein2024 | Maier-Hein L, Reinke A, Godau P, et al. Metrics reloaded: recommendations for image analysis validation. Nat Methods. 2024;21(2):195-212. | doi:10.1038/s41592-023-02151-z | metric selection |
| reinke2024 | Reinke A, Tizabi MD, et al. Understanding metric-related pitfalls in image analysis validation. Nat Methods. 2024;21(2):182-194. | doi:10.1038/s41592-023-02150-0 | pitfalls of averaging, empty references, size dependence |
| taha2015 | Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. BMC Med Imaging. 2015;15:29. | doi:10.1186/s12880-015-0068-x | metric definitions (Dice, HD95, ASSD) |
| varoquaux2022 | Varoquaux G, Cheplygina V. Machine learning for medical imaging: methodological failures and recommendations for the future. npj Digit Med. 2022;5:48. | doi:10.1038/s41746-022-00592-y | validation-set overfitting, small-sample optimism |
| kapoor2023 | Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns. 2023;4(9):100804. | doi:10.1016/j.patter.2023.100804 | data leakage taxonomy; inflated results |
| eelbode2020 | Eelbode T, Bertels J, Berman M, et al. Optimization for medical image segmentation: theory and practice when evaluating with Dice score or Jaccard index. IEEE Trans Med Imaging. 2020;39(11):3679-3690. | doi:10.1109/TMI.2020.3002417 | metric-sensitive (Dice) losses outperform cross-entropy when evaluated by Dice |
| yeung2022 | Yeung M, Sala E, Schönlieb C-B, Rundo L. Unified focal loss: generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation. Comput Med Imaging Graph. 2022;95:102026. | doi:10.1016/j.compmedimag.2021.102026 | loss-based class-imbalance handling |
| salehi2017 | Salehi SSM, Erdogmus D, Gholipour A. Tversky loss function for image segmentation using 3D fully convolutional deep networks. MLMI 2017, LNCS 10541:379-387. | arXiv:1706.05721 | Tversky loss |
| chlap2021 | Chlap P, Min H, Vandenberg N, Dowling J, Holloway L, Haworth A. A review of medical image data augmentation techniques for deep learning applications. J Med Imaging Radiat Oncol. 2021;65(5):545-563. | doi:10.1111/1754-9485.13261 | taxonomy of CT/MRI augmentation |
| shorten2019 | Shorten C, Khoshgoftaar TM. A survey on image data augmentation for deep learning. J Big Data. 2019;6:60. | doi:10.1186/s40537-019-0197-0 | general augmentation survey |
| wilcoxon1945 | Wilcoxon F. Individual comparisons by ranking methods. Biometrics Bull. 1945;1(6):80-83. | doi:10.2307/3001968 | signed-rank test |
| holm1979 | Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat. 1979;6(2):65-70. | JSTOR 4615733 | Holm correction |
| page1963 | Page EB. Ordered hypotheses for multiple treatments: a significance test for linear ranks. J Am Stat Assoc. 1963;58(301):216-230. | doi:10.1080/01621459.1963.10500843 | trend test |
| cliff1993 | Cliff N. Dominance statistics: ordinal analyses to answer ordinal questions. Psychol Bull. 1993;114(3):494-509. | doi:10.1037/0033-2909.114.3.494 | Cliff's delta |
| denotter | DenOtter TD, Schubert J. Hounsfield unit. StatPearls. NCBI Bookshelf. | NBK547721 | HU definition only |
| tfbinaryiou | TensorFlow. tf.keras.metrics.BinaryIoU (API documentation). | tensorflow.org | default target_class_ids=(0, 1) averages background and lesion IoU |

Items marked **Preprint** are not peer-reviewed. The INSTANCE overview is cited only for
facts about the challenge (dataset sizes, ranking metrics, top Dice); if a peer-reviewed
version appears before resubmission, replace the citation.
