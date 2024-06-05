# Background

The UK is presnetly running a series of lung cancer screening (LCS) pilots as part of an aim to rollout LCS nationally. All  of the LCS pilots utilise Computer-Aided Detection (CADe) to assist Radiologists in detecting potential abnormalities. Contemporary CADe systems predominantly use deep learning (DL) models, which require substantial amounts of data. Research has demonstrated that DL models struggle to generalise on out-of-domain data. To safeguard their intellectual property, companies developing commercial Computer-Aided Detection (CADe) systems maintain confidentiality regarding both the architecture of their models and the specifics of their training data. 

# Objective

The aim of this project was to investigate how well state-of-the-art nodule detection algorithms, trained on public datasets generalise to a dataset representative of a lung cancer screening dataset.

# Methods

Two state-of-the-art nodule detection algorithms were selected based having different algorithmic appraoches as well as having high peformance on a popular nodule detection dataset, LUNA16. LUNA16 is the de-facto dataset that nodule detection algorithms use when demonstrating performance. LUNA16 consists of 888 lung ct scans and has 1186 nodule locations recorded, which can used as the ground truths for model development.  The FROC was produced with 1000 bootstraps. These models, having been trained on LUNA were then validated on 500 ct scans drawn from the SUMMIT study, which was a recent lung cancer screening study.

Two state-of-the-art nodule detection algorithms were selected based having different algorithmic appraoches as well as having high peformance on a popular nodule detection dataset, LUNA16. LUNA16 consists of 888 CT scans and has 1186 nodule locations and diameteres recorded. he selected nodule detection algorithms underwent initial training and evaluation on the LUNA dataset, employing a 5-fold cross-validation training strategy. The representative LCS dataset consisted of 595 CT scans randomly drawn from the SUMMIT study, with 767 nodule locations recorded. The free operating receiver curve (FROC) is a popular metric for evaluating the performance of nodule detection algorithms. The FROC curve measures senstivity at different operating points i.e., number of false postitives per scan. This can be condensed into a single metric called the competition performance metric (CPM), which is defined as the average sensitivity measured at the following operating points: 1/8, 1/4, 1/2, 1, 2, 4 and 8 false positives per scan.

# Results

The results of the two models, model 1 had a CPM of (0.88 CI: +- 20) and model 2 had a CPM of (099 +- 1). When evaluating the models trained on the LUNA dataset on the SUMMIT dataset, model 1 had a CPM of (0.45 CI: +- 20) and model 2 had a CPM of (0.66 CI: +- 20). When looking at the severity of the nodules missed by each model, model 1 missed 5% of actionable nodules whereas model 2 missed 4% of actionable nodules.


# Conclusions

This discrepancy in sensitivity metrics underscores the challenges associated with the algorithm's generalization across diverse populations and imaging conditions. Our findings highlight the critical importance of thorough validation on region-specific datasets, emphasizing the need for continued refinement and customization of detection algorithms to ensure their effectiveness in real-world clinical scenarios. Further investigation into the potential biases within the training data is warranted to enhance the algorithm's adaptability and reliability across varied clinical contexts.