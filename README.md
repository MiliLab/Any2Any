
<div align="center">

<h1>Any2Any: Unified Arbitrary Modality Translation for Remote Sensing</h1>

Haoyang Chen<sup>1,2</sup>, 
Jing Zhang<sup>1,2 †</sup>, 
Hebaixu Wang<sup>1,2</sup>, 
Shiqin Wang<sup>1</sup>, 
PoHsun Huang<sup>1</sup>, 
Jiayuan Li<sup>2,3</sup>, 
Haonan Guo<sup>1,2</sup>, 
Di Wang<sup>1,2 †</sup>, 
Zheng Wang<sup>1,2 †</sup>, 
Bo Du<sup>1,2 †</sup>.

<sup>1</sup> Wuhan University,  <sup>2</sup> Zhongguancun Academy,  <sup>3</sup> Beijing Institute of Technology.

<sup>†</sup> Corresponding author

</div>

<p align="center">
  <a href="#-update">News</a> |
  <a href="#-abstract">Abstract</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-checkpoints">Checkpoints</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-statement">Statement</a>
</p >


## 🔥 Update

**2025.11.25**
- The paper is post on arXiv **([arXiv 2603.04114](https://arxiv.org/abs/2603.04114))** 

## 🌞 Abstract

Multi-modal remote sensing imagery provides complementary observations of the same geographic scene, yet such observations are frequently incomplete in practice. Existing cross-modal translation methods treat each modality pair as an independent task, resulting in quadratic complexity and limited generalization to unseen modality combinations. We formulate Any-to-Any translation as inference over a shared latent representation of the scene, where different modalities correspond to partial observations of the same underlying semantics. Based on this formulation, we propose **Any2Any**, a unified latent diffusion framework that projects heterogeneous inputs into a geometrically aligned latent space. Such structure performs anchored latent regression with a shared backbone, decoupling modality-specific representation learning from semantic mapping. Moreover, lightweight target-specific residual adapters are used to correct systematic latent mismatches without increasing inference complexity. To support learning under sparse but connected supervision, we introduce **RST-1M**, the first million-scale remote sensing dataset with paired observations across five sensing modalities, providing supervision anchors for any-to-any translation. Experiments across 14 translation tasks show that Any2Any consistently outperforms pairwise translation methods and exhibits strong zero-shot generalization to unseen modality pairs.

<figure>
<div align="center">
<img src=Images/figure1.png width="60%">
</div>

<div align='center'>

**Figure 1. Quantitative comparison between our proposed Any2Any and other compar methods.**

</div>

## 📖 Datasets

<div align="center">
<img src=Images/figure2.png width="60%">
</div>

<div align='center'>

**Figure 2. Statistics and example images of the RST-1M dataset.**

</div>

The RST-1M is Coming Soon.

## 🚀 Model

<div align="center">
<img src=Images/figure3.png width="100%">
</div>

<div align='center'>

**Figure 3. Overview of the Any2Any framework.**

</div>

The checkpoints are Coming Soon.

## 🔨 Usage

### Training

Wait for update.

### Inference

We provide an inference script:

Wait for update.

## 🍭 Results


<div align="center">
<img src=Images/figure4.png width="100%">
</div>
<div align='center'>

**Figure 4. Qualitative comparison between our proposed Any2Any and other compar methods.**
</div>

<div align="center">
<img src=Images/figure5.png width="60%">
</div>
<div align='center'>

**Figure 5. Qualitative results of our method on unseen remote sensing modality translation tasks with missing paired training data.**
</div>

## ⭐ Citation

If you find Any2Any helpful, please give a ⭐ and cite it as follows:

```

@article{chen2026Any2Any,
  title={Any2Any: Unified Arbitrary Modality Translation for Remote Sensing},
  author={Chen, Haoyang and Jing, Zhang and Wang, Hebaixu and Wang, Shiqin and Huang, Pohsun and Li, Jiayuan and Guo, Haonan and Wang, Di and Wang, Zheng and Du, Bo},
  journal={arXiv preprint arXiv:2603.04114},
  year={2026}
}
```

## 🎺 Statement

For any other questions please contact Haoyang Chen at [whu.edu.cn](mailto:haoyangchen@whu.edu.cn).


