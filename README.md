# Unified-Multi-lesion-Segmentation\nThis is an end-to-end framework for multi-lesion segmentation of fundus images, originally proposed by aran085. For insight into our approach, please refer to our [paper](https://doi.org/10.1016/j.neucom.2019.04.019).\n### Introduction:\nDiabetic retinopathy and diabetic macular edema are leading causes for blindness in working-age people. Diagnosis usually relies on the quantitative and qualitative analysis of lesions in fundus images. However, segmenting these lesions can be challenging due to uncertainties in size, contrast, and high interclass similarity. Hence, our aim was to develop a model that can accurately segment various types of lesions.\nOur solution was the Unified-Multi-lesion-Segmentation, a small object segmentation network that processes different types of lesions. It introduces a multi-scale feature fusion strategy to tackle small lesion regions and a multi-channel bin loss to manage class-imbalance and loss-imbalance problems.\nWe have performed evaluations on 