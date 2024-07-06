---
layout: distill
title: "Hydra Part II - The Model"
description: 
tags:
giscus_comments: false
date: 2024-07-06
featured: false
thumbnail: assets/img/2024-07-06-hydra/logo_trans.png

authors:
  - name: Sukjun Hwang*
    url:
    affiliations:
      name: CMU
  - name: Aakash Lahoti*
    url:
    affiliations:
      name: CMU
  - name: Ratish Puduppully
    url:
    affiliations:
      name: A*STAR
  - name: Tri Dao
    url:
    affiliations:
      name: Princeton
  - name: Albert Gu
    url:
    affiliations:
      name: CMU, Cartesia AI

bibliography: june.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: SSMs Are Semiseparable Matrix Mixers
  - name: Quasiseparable Matrices
    subsections:
      - name: Quasiseparable Matrices $\supset$ Semiseparable and Low-Rank Matrices
      - name: Quasiseparable Matrices $\supset$ Two Separate SSMs
  - name: Hydra
    subsections:
      - name: Implementation
      - name: Performance
  - name: Epilogue

---

{% include figure.liquid loading="eager" path="assets/img/2024-07-06-hydra/logo_trans.png" %}

[[Paper](https://arxiv.org/abs/TODO)]
[[Code](https://github.com/goombalab/hydra)]


1. [Part I - Matrix Mixer Framework]({% post_url 2024-07-06-hydra-part1-matrix-mixer %})
2. Part II - Hydra: The Model

In our previous post, we systematically compared various sequence models with different mixer matrices, and the quasiseparable SAM mixer emerged as the top performer. So, what exactly is it?

## Recap: SSMs Are Semiseparable Matrix Mixers

Before diving into the details of quasiseparable SAM mixers, let’s briefly revisit some key findings from [Mamba-2](https://arxiv.org/abs/2405.21060)<d-cite key="ssd"></d-cite>. Recently, Mamba-2 has shown that the mixer matrices of SSMs are inherently parametrized to one of the fundamental structured matrix classes -- semiseparable matrices.

> **Defintion** of Semiseparable Matrices \\
> A lower triangular matrix $\textbf{M}$ is $N$-semiseparable iff any submatrix from the lower triangle (on or below the diagonal) has a rank of at most $N$. See (a) in the figure below.

So why are SSMs semiseparable matrix mixers? Using our previously defined matrix mixer framework, we can represent SSMs as follows:

$$
\begin{align}
    \textbf{y}_t &= \sum^{t}_{s=0} \textbf{C}^T_t \left(\prod_{k=s+1}^{i} \textbf{A}_{k}\right) \textbf{B}_s \textbf{x}_s \\
    \\
    \textbf{Y} &= \text{SSM}(\textbf{A}, \textbf{B}, \textbf{C})(\textbf{X}) = \textbf{M} \textbf{X} \space ,\\
    \\
    m_{ij} & = \textbf{c}^T_i \textbf{A}_i \cdots \textbf{A}_{j+1} \textbf{b}_j
\end{align}
$$

where each matrix $\textbf{A}_i \in \mathbb{R}^{N \times N}$ and vector $\textbf{c}_i, \textbf{b}_i \in \mathbb{R}^{N \times 1}$. This decomposition shows that SSMs are indeed semiseparable mixers. [If you are not familiar with this concept, we recommend checking out this [blog post]({% post_url 2024-05-31-mamba2-part2-theory %}) for a great explanation.]

Semiseparable matrices are an excellent choice for mixer matrices -- they are sub-quadratic, performant, and can be extended to handle sequences of various lengths. However, there’s one significant limitation: due to their definition, the upper right triangle of semiseparable matrices is filled with zeros, making them inevitably causal. This limitation makes SSMs incapable of **bidirectional sequence processing**.

Why is bidirectionality important? Bidirectional processing is crucial for several reasons. One major reason is its importance in handling multiple modalities, such as processing 2D images. Without bidirectionality, models can’t fully leverage information from both past and future contexts within a sequence, which is essential for comprehensive data analysis across various applications.

A straightforward way to make SSMs bidirectional is to use two separate SSMs: one for forward sequence processing and one for reverse sequence processing. There are several approaches to combine their outputs, such as adding, multiplying, or concatenating them <d-cite key="sashimi"></d-cite><d-cite key="vision_mamba"></d-cite><d-cite key="caduceus"></d-cite><d-cite key="bigs"></d-cite><d-cite key="mssm"></d-cite>. While these heuristics can work, they lack a principled design philosophy, leading to different heuristics being used for different tasks without a systematic approach.

But what if we could use the matrix mixer framework to systematically derive the optimal $\textbf{M}$? Absolutely, we can! In addition to the three desiderata we discussed previously -- sub-quadratic complexity, extendability, and high-performance -- let’s add one more requirement: **bidirectionality**. For the mixer matrix to achieve bidirectionality, it must feature upper triangular components. So, how should we fill them?

{% include figure.liquid loading="eager" path="assets/img/2024-07-06-hydra/semiquasi_trans.png" %}

## Structured Matrix of Our Choice: Quasiseparable Matrices

For our bidirectional sequence mixer, we choose quasiseparable matrices. So, what makes quasiseparable matrices stand out? Let’s start by looking at their definition.

> **Defintion** of Quasiseparable Matrices by the Rank Characterization. \\
> A matrix $\textbf{M}$ is $N$-quasiseparable iff any submatrix from either the strictly upper or lower triangle (off from the diagonal) has a rank of at most $N$. See (b) in the figure above.

At first glance, this definition might seem similar to that of semiseparable matrices. To clarify, let’s highlight the key differences between quasiseparable and semiseparable matrices:

|  | **Semiseparable** | **Quasiseparable** |
|--|---|---|
| (I) | any submatrix from *the lower triangle* | any submatrix from either the strictly *upper or lower triangle* |
| (II) | *on or below* the diagonal          | *off* from the diagonal                                      |

### Quasiseparable Matrices $\supset$ Semiseparable and Low-Rank Matrices

Although the differences between quasiseparable and semiseparable matrices might seem subtle, they lead to significant improvements in expressivity. According to difference **(I)**, semiseparable matrices zero out the upper triangular elements, while quasiseparable matrices extend to include these elements, enabling bidirectionality. Consequently, semiseparable matrices can only generalize mixers that use causal low-rank matrices, such as Linear Attention, whereas quasiseparable matrices generalize typical low-rank matrices. Moreover, both differences **(I)** and **(II)** mean that quasiseparable matrices not only generalize but also extend semiseparable matrices.

- ***Quasiseparable matrices generalize low-rank matrices.***
- ***Quasiseparable matrices generalize and extend semiseparable matrices.***

### Quasiseparable Matrices $\supset$ Two Separate SSMs

We now understand that for bidirectional processing scenarios, quasiseparable mixers are indeed better than semiseparable matrices. But what makes quasiseparable mixers superior to the bidirectional extensions using two separate SSMs?

Heuristic variants that use the Hadamard product and concatenation <d-cite key="bigs"></d-cite><d-cite key="mssm"></d-cite> are difficult to analyze systematically within the matrix mixer framework. Moreover, concatenation variants double the number of output channels, necessitating additional parameters for reducing the number of channels.

In contrast, addition-based variants <d-cite key="sashimi"></d-cite><d-cite key="vision_mamba"></d-cite><d-cite key="caduceus"></d-cite> can be formulated using the matrix mixer framework, as shown in \(c\) of the figure above, which resembles quasiseparable matrices in (d). However, difference **(II)** highlights that the diagonals of semiseparable matrices are also constrained by the rank characterization, and consequently, so are the diagonals of addition-based extensions. Quasiseparable matrices, on the other hand, do not have this constraint on the diagonals, allowing them to be complete free parameters. This flexibility makes quasiseparable matrices more mathematically expressive than addition-based bidirectional extensions.

- ***Quasiseparable matrices are strictly more expressive than mixer matrices of addition-based bidirectional SSMs.***

This property of complete freedom in the diagonals of quasiseparable matrices is more evident in another definition of quasiseparable matrices:

> A matrix $\textbf{M}$ is $N$-quasiseparable if each element $m_{ij}$ satisfies:
>
> $$\begin{equation}
>   m_{ij} =
>       \begin{cases}
> \overrightarrow{\textbf{c}^{T}_{i}} \overrightarrow{\textbf{A}^{\times}_{i:j}} \overrightarrow{\textbf{b}_{j}},  & \text{if } i > j \\
> \delta_{i},         & \text{if } i = j \\
> \overleftarrow{\textbf{c}^{T}_{j}} \overleftarrow{\textbf{A}^{\times}_{j:i}} \overleftarrow{\textbf{b}_{i}},  & \text{if } i < j\\
>       \end{cases},\\
> \end{equation}
> $$ 
>
> where each $\delta_i$ is a scalar, $\textbf{b}_i, \textbf{c}_i \in \mathbb{R}^{N \times 1}$, and $\textbf{A}_i \in \mathbb{R}^{N \times N}$. 

These are the actual results we obtained for the C4 and GLUE benchmark, along with the validation loss curve. Supported by these theoretical claims, our Hydra model, which uses a quasiseparable mixer matrix, indeed has shown superior performance to previous heuristic bidirectional extensions!

{% include figure.liquid loading="eager" path="assets/img/2024-07-06-hydra/bidirectionality_trans.png" %}

## Hydra: Our Main Bidirectional Sequence Mixer

### Implementation

Now that we've confirmed quasiseparable matrices as the go-to mixer matrices, we fully leverage them to propose the two-headed Mamba -- ***Hydra***. Take a look at part (d) in the figure above, which illustrates the mixer matrix of Hydra, and also notice it's also our previosly defined SAM! Utilizing an SSM, which is a semiseparable mixer, we can implement Hydra with the following formula:
$$
QS(\textbf{X}) = \texttt{shift}(SS(\textbf{X})) + \texttt{flip}(\texttt{shift}(SS(\texttt{flip}(\textbf{X})))) + \textbf{DX},
$$ where $\textbf{X}$ is the input sequence, $\texttt{flip}(\cdot)$ denotes a function that reverses the input, $\texttt{shift}(\cdot)$ denotes a right-shift function, and $\textbf{D} = \text{diag}(\delta_1, \cdots, \delta_L)$ represents the diagonal elements of $QS$. Here, $QS(\cdot)$ and $SS(\cdot)$ are the mixer matrix of Hydra and an SSM, respectively. 

Among the various iterations of SSMs, we adopt the latest one -- SSD from Mamba-2. Since SSMs are sub-quadratic, this simple implementation maintains the sub-quadratic cost. Compared to heuristic extensions that use two separate SSMs for bidirectionality, Hydra shares the input processing function $f_X$ for forward and reverse sequence processing, which nearly halves the number of parameters.

You can check out [the actual code](https://github.com/goombalab/hydra_private/blob/main/hydra/modules/hydra.py). To sum up:
- Hydra's matrix mixer is meticulously parameterized to be a quasiseparable matrix with enhanced expressivity through shift operations.
- Hydra is sub-quadratic and super easy to implement using existing SSM implementations like Mamba.
- Hydra greatly reduces parameter counts compared to bidirectional extensions using two SSMs.

### Performance

We have seen that Hydra outperforms heuristic bidirectional extensions of SSMs, but how does it compare to state-of-the-art methods? Surprisingly, Hydra surpasses all previous models, including Transformer-based models such as BERT and ViT. When matched for the number of parameters, Hydra consistently shows the best performance across both NLP and Vision domains, highlighting its versatility.

<table>
    <tr>
        <td colspan="3" style='font-weight:bold; text-align:center; background-color: #4dabf7'>NLP</td>
        <td colspan="3" style='font-weight:bold; text-align:center; background-color: #69db7c'>Vision</td>
    </tr>
    <tr>
        <td style='font-weight:bold;'>Method</td>
        <td style='font-weight:bold;'># Params</td>
        <td style='font-weight:bold;'>GLUE Avg</td>
        <td style='font-weight:bold;'>Method</td>
        <td style='font-weight:bold;'># Params</td>
        <td style='font-weight:bold;'>Top-1 (%)</td>
    </tr>
    <tr>
        <td style='font-weight:bold;'>BERT<d-cite key="bert"></d-cite></td>
        <td>110M</td>
        <td>83.5</td>
        <td style='font-weight:bold;'>ViT-B<d-cite key="vit"></d-cite></td>
        <td>87M</td>
        <td>78.8</td>
    </tr>
    <tr>
        <td style='font-weight:bold;'>MLP-Mixer<d-cite key="mlpmixer"></d-cite></td>
        <td>112M</td>
        <td>77.5</td>
        <td style='font-weight:bold;'>S4-ViT-B<d-cite key="s4"></d-cite><d-cite key="s4d"></d-cite></td>
        <td>89M</td>
        <td>79.4</td>
    </tr>
    <tr>
        <td style='font-weight:bold;'>FNet<d-cite key="fnet"></d-cite></td>
        <td>112M</td>
        <td>75.8</td>
        <td style='font-weight:bold;'>Hyena-ViT-B<d-cite key="hyena"></d-cite></td>
        <td>88M</td>
        <td>78.4</td>
    </tr>
    <tr>
        <td style='font-weight:bold;'>M2<d-cite key="m2"></d-cite></td>
        <td>116M</td>
        <td>80.9</td>
        <td style='font-weight:bold;'>Mamba-ViT-B<d-cite key="mamba"></d-cite><d-cite key="ssd"></d-cite></td>
        <td>89M</td>
        <td>79.1</td>
    </tr>
    <tr>
        <td style='background-color: #f783ac; font-weight:bold;'>Hydra</td>
        <td>112M</td>
        <td>84.3</td>
        <td style='background-color: #f783ac; font-weight:bold;'>Hydra-ViT-B</td>
        <td>91M</td>
        <td>81.0</td>
    </tr>
</table>

On the GLUE benchmark, Hydra outperforms BERT by 0.8 points. On ImageNet-1K, Hydra improves by 2.2 points over ViT. These results underscore Hydra’s capability to set new standards in both natural language processing and image classification tasks!

## Epilogue

Lately, the demand for large-scale computation has never been higher. Since the emergence of Mamba, interests in structured matrices has surged, and now is their time to shine. Structured matrices offer an exciting approach to efficient and powerful input processing, similar to how M2 improved over MLP-Mixer.

In recent years, we’ve seen numerous groundbreaking works showcasing promising results using structured matrices like Mamba. If the community strives together, just as we have spent about seven years investigating and improving Transformers, we believe there is enormous potential for further advancements through systematic exploration of different structured matrices, along with better optimized training settings (which have been fine-tuned for Transformers).

A big shout-out to the recent [BTT](https://arxiv.org/abs/2406.06248)<d-cite key="btt"></d-cite> work, which systematically explores structured matrices for effective channel mixers. We were very excited to see this kind of systematic investigation, which is crucial for the continued advancement of better architectures.
