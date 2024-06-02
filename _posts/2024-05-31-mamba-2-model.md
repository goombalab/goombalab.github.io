---
layout: distill
title: State Space Duality (Mamba-2) Part 1 - The Model
description: 
tags:
giscus_comments: true
date: 2024-05-31
featured: true

authors:
  - name: Albert Gu
    url:
    affiliations:
      name: Carnegie Mellon University
  - name: Tri Dao
    url:
    affiliations:
      name: Princeton

bibliography: albert.bib

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
  - name: The SSD Model
    subsections:
      - name: The Linear (SSM) Mode
      - name: The Quadratic (Attention) Mode
  - name: State Space Duality
    subsections:
      - name: SSD vs. State Space Models
      - name: SSD vs. Attention
  - name: Best of Both Worlds
  - name: The Mamba-2 Architecture

---

{% include figure.liquid loading="eager" path="assets/img/2024-05-31-mamba-2/mamba-2-V3-transparent.png" %}

[Paper]

[Code](https://github.com/state-spaces/mamba)

Since the release of [Mamba](https://arxiv.org/abs/2312.00752) 6 months ago, we've been pleasantly surprised by the overwhelming [community response](https://github.com/AvivBick/awesome-ssm-ml).
It's been incredibly gratifying to see the line of research on efficient sequence models we've been pursuing for years really resonate with the machine learning community and take off more than we could have anticipated.
We've seen an enormous amount of exciting follow-up work, from direct applications
(e.g. vision <d-cite key="zhu2024vision"></d-cite><d-cite key="ma2024u"></d-cite><d-cite key="liu2024vmamba"></d-cite>, genomics <d-cite key="schiff2024caduceus"></d-cite>, graphs <d-cite key="wang2024graph"></d-cite><d-cite key="behrouz2024graph"></d-cite>, and more)
to understanding (e.g. on recall abilities <d-cite key="jelassi2024repeat"></d-cite>,
in-context learning<d-cite key="akyurek2024context"></d-cite> <d-cite key="grazzi2024mamba"></d-cite> <d-cite key="park2024can"></d-cite>,
and formal language expressivity <d-cite key="merrill2024illusion"></d-cite><d-cite key="sarrof2024expressive"></d-cite>),
and an enormous number of [online](https://jackcook.com/2024/02/23/mamba.html) [blogs](https://srush.github.io/annotated-mamba/hard.html),
[tutorials](https://www.youtube.com/watch?v=dVH1dRoMPBc),
[and](https://www.youtube.com/watch?v=8Q_tqwpTpVU)
[videos](https://www.youtube.com/watch?v=N6Piou4oYx8).
We couldn't be more excited about the direction of this research!


Yet despite its potential so far, we weren't completely satisfied with the first version of Mamba...

### Problem 1 (Understanding)
From a conceptual standpoint, one of the reasons we found SSMs so fascinating is how they just feel _fundamental_. One way this is exemplified is how they have rich ties to many major paradigms of sequence models.
As developed in our earlier works on structured SSMs <d-cite key="gu2021combining"></d-cite><d-cite key="gu2023thesis"></d-cite>, they seem to capture the essence of continuous, convolutional, and recurrent sequence models -- all wrapped up in a simple and elegant model.

But of course, aside from these, there is another major sequence model paradigm: the variants of the ubiquitous **attention** mechanism<d-cite key="bahdanau2015neural"></d-cite><d-cite key="vaswani2017attention"></d-cite>.
SSMs always felt somewhat disjoint from attention, and we've tried for a while to understand their relationship better.

> Question 1: **What are the conceptual connections between state space models and attention?** Can we combine them?

### Problem 2 (Efficiency)
From a computational standpoint, 
despite the work that went into making Mamba fast (in particular, its hardware-aware selective scan implementation) it is still much less hardware-efficient than mechanisms such as attention.
The missing piece is that modern accelerators such as GPUs and TPUs are *highly* specialized for matrix multiplications.
While this is not a problem for inference, which is bottlenecked by somewhat different considerations, this can be a big deal during training time.
[//]: # For example, an end-to-end Mamba-1 model is XX times slower than an equivalent Transformer.

> Question 2: **Can we speed up the training of Mamba models by recasting them as matrix multiplications?**

These are the main questions that Mamba-2 -- in particular, its new state space model variant -- tries to address.


## The SSD Model

The main point of the Mamba-2 paper is what we call **structured state space duality** (SSD),
which refers to both
1. a specific model, or more precisely a standalone layer like an SSM or attention that can be incorporated into deep neural networks.
2. a general framework for reasoning about this model (and beyond).

The main "state space dual model" itself really isn't so scary!
In this first part of a series of blog posts, we'll provide a self-contained description of the SSD layer (and Mamba-2) in isolation and how it compares to related models, particularly Mamba-1.

In [the next part], we'll describe the general framework and theoretical connections, which aren't necessary to actually use Mamba-2.

### The Linear (SSM) Mode

SSD starts from the same set of equations as Mamba:

$$
\begin{aligned}
h_{t} &= A_t h_{t-1} + B_t x_t \\
y_t &= C_t^{\top} y_t
\end{aligned}
$$

\begin{equation}
\label{eq:ssm}
(\text{Selective state space model (SSM)})
\end{equation}


To recap, a **structured state space model (SSM)** <d-cite key="gu2022efficiently"></d-cite><d-cite key="gu2023thesis"></d-cite> defines a map from $x \in \mathbb{R}^\mathtt{T} \to y \in \mathbb{R}^\mathtt{T}$.
Think of $x_t$ and $y_t$ as being scalars, and the hidden state $h_t$ as an $\mathtt{N}$-dimensional vector, where $\mathtt{N}$ is an independent hyperparameter called the *state size, state dimension, or state expansion factor*.

A *selective* state space model allows the $(A, B, C)$ SSM parameters to vary across time <d-cite key="gu2023mamba"></d-cite>.
We'll think of them as tensors with shapes $A \in \mathbb{R}^\mathtt{(T, N, N)}$, $B \in \mathbb{R}^\mathtt{(T, N)}$, and $C \in \mathbb{R}^\mathtt{(T, N)}$ respectively.<d-footnote>As with Mamba-1, we take everything over the reals $\mathbb{R}$, although complex variants as with other structured SSMs are also possible.</d-footnote>

Structured SSMs require $A$ to have structure to be efficiently computable, such as the most commonly used diagonal structure <d-cite key="gu2022parameterization"></d-cite><d-cite key="gupta2022diagonal"></d-cite><d-cite key="smith2023s5"></d-cite><d-cite key="gupta2022simplifying"></d-cite>.
In this case $A$ has shape $\mathtt{(T, N)}$ where only the diagonal elements of the $\mathtt{N} \times \mathtt{N}$ matrices are stored.

#### SSD: Scalar Structured SSM
The original Mamba (or more precisely its core "S6" layer) is exactly a selective SSM with diagonal structure. 

**The SSD layer of Mamba-2 makes only one small modification**: it restricts the diagonal $A$ even further to a *scalar times identity* structure; in other words the diagonal elements of $A$ must all be the same value.
In this case $A$ can be represented with shape just $\mathtt{(T)}$ and one can also identify $A_t$ as just a scalar (and so we'll sometimes denote it $a_t$).

#### Multihead SSMs

Equation \eqref{eq:ssm} is defined only for a single dimensional input $x \in \mathbb{R}^\mathtt{T}$.
If $X \in \mathbb{R}^\mathtt{(T, P)}$ has $\mathtt{P}$ separate channels,
we can use the same dynamics (i.e. the same SSM $(A, B, C)$) independently for each channel.
This can be interpreted as a *single head* of the SSM model.

Here, we think of $X$ as a tensor of shape $\mathtt{(T, P)}$ where $\mathtt{T}$ is the sequence (time) dimension and $\mathtt{P}$ is the "head dimension".<d-footnote>Normally there's an additional batch dimension $\mathtt{B}$ when implementing these models, which we'll ignore throughout this presentation.</d-footnote>

Multiple heads can be constructed completely independently;
for the remainder of this post, we assume that we're working with a single head.
Note that these heads are exactly analogous to how heads in multi-head attention models work,
and in Mamba-2 we also choose similar dimensions as modern Transformers, e.g. $\mathtt{P} = 64$ or $\mathtt{P}=128$.

We can notate the general (selective) state space model as
\begin{equation}
\label{eq:ssm-transformation}
Y^\mathtt{(T,P)} = \mathsf{SSM}(A^\mathtt{(T,...)}, B^\mathtt{(T,N)}, C^\mathtt{(T,N)})(X^\mathtt{(T,P)})
\end{equation}

Some axes of variation include
1. The structure on $A$, which affects its parameter shape, e.g. `... = (N)` for diagonal SSMs, or `... = ()` for scalar SSMs (i.e. SSD)
2. The state dimension $\mathtt{N}$ (i.e. `d_state`)
3. The head dimension $\mathtt{P}$ (i.e. `d_head`)

There are other axes of variation of structured SSMs (e.g. time-invariance vs. selectivity, SISO vs. MIMO<d-cite key="smith2023s5"></d-cite>, real vs. complex, etc.),
but we're highlighting these so that we can contrast Mamba-2 to Mamba-1 in just a second...


### The Quadratic (Attention) Mode

Let's switch tacks and forget about state space models for a moment.
Given the same tensors above with the same shapes $(A^\mathtt{(T)}, B^\mathtt{(T, N)}, C^\mathtt{(T, N)})$,
let's define a different object.

First, we'll define the following matrix (don't worry, we'll explain more and give it a name in the second part of this post).

$$
  L =
  \begin{bmatrix}
    1 & \\
    a_1 & 1 & \\
    a_2a_1 & a_2 & 1 \\
    \vdots & \vdots & \ddots & \ddots \\
    a_{\mathtt{T}-1}\dots a_1 & a_{\mathtt{T}-1}\dots a_2 & \dots & a_{\mathtt{T}-1} & 1 \\
  \end{bmatrix}
  .
$$

Then, let's define the following matrix

\begin{equation}
\label{eq:ssd-attention}
M = L \circ C B^\top \in \mathbb{R}^{\mathtt{(T,T)}}
\end{equation}

Finally, $M$ encodes a *sequence transformation*
$x \in \mathbb{R}^\mathtt{T} \to y \in \mathbb{R}^\mathtt{T}$
mapping a 1D input to a 1D output---just as in equation \eqref{eq:ssm}---through basic matrix multiplication $y = Mx$.

What's special about this?
Well, you may notice that it looks very similar to an attention computation.
In fact, if all $a_t = 1$, then $L$ is simply the lower-triangular *causal mask* and \ref{eq:ssd-attention} is exactly **causal linear attention** <d-cite key="katharopoulos2020transformers"></d-cite>:

$$
Y = (L \circ Q K^\top) V
$$

This is exactly the same as equation \eqref{eq:ssd-attention} if we rename $(C, B, X) \mapsto (Q, K, V)$!

## State Space Duality

The so-called "duality" refers to the fact that the two models defined in equations \eqref{eq:ssm} (for the scalar-identity structured $A_t$ case) and \eqref{eq:ssd-attention} are actually *exactly the same model*, which we can view as a particular function

$$
(A^\mathtt{(T)}, B^\mathtt{(T, N)}, C^\mathtt{(T, N)}, X^\mathtt{(T, P)}) \mapsto Y^\mathtt{(T, P)}
$$

with tensor shapes specified above.

In the general *SSD Framework*, we'll show this equivalence in two completely different ways, both of which are actually much more general and each quite illuminating.

If you take our word for it, though, then SSD is relatively simple to contrast in relation to either SSMs or attention.

### SSD vs. State Space Models
Compared to previous SSMs, SSD is pretty much the same as the core layer of Mamba but with even more structure on the recurrent $A$ matrices.
1. Mamba-1 (S6) uses diagonal structure on $A$, while Mamba-2 (SSD) uses scalar-times-identity structure on $A$.
2. Mamba-1 has a head dimension of $\mathtt{P}=1$ (i.e. all channels are completely independently controlled), while Mamba-2 uses a head dimension of $\mathtt{P}>1$ (something like $\mathtt{P}=64$ by default).

In particular, this can be viewed as weight-tied in two ways:
- By restricting the diagonal structure of $A$ to scalar-times-identity, the scalar recurrence dynamics are tied across all $\mathtt{N}$ elements of the state space.
- These dynamics are also shared across all $\mathtt{P}$ channels of a given head.

In other words, a single SSM head has total state size $\mathtt{P} \times \mathtt{N}$,
which are each governed by separate scalar recurrences in Mamba but are controlled by a single shared recurrence in Mamba-2.

Why make these restrictions? The main motivation is efficiency: these changes are necessary to be able to view the model in its [[dual attention form](#the-quadratic-attention-mode)], which allows matrix multiplications to be used.

> #### The Bottom Line: Mamba-1 vs. Mamba-2
>
> Compared to Mamba-1, Mamba-2 allows **much larger state dimensions** (from `N=16` in Mamba-1 to `N=64` to `N=256` or even higher in Mamba-2) while simultaneously being **much faster during training**.
{: .block-tip}

But can this hurt us? There's some intuition to believe that it shouldn't.
One of the main reasons for the selectivity (e.g. $A$ that depends on the input $X$) introduced in Mamba
is to let the SSM be able to control whether to remember or ignore particular pieces of information;
for example, if a filler "um" is encountered in a text transcript.
But if such information should be ignored, then the entire state can ignore it together, and so it should be okay if the state's dynamics are shared across all features.

Empirically, we haven't found evidence that the restricted expressivity of Mamba-2 might hurt, but the jury's still out!
From one perspective, Mamba-2 isn't *strictly* better than Mamba-1: while it's a dramatic improvement from a *training* perspective, Mamba-1 might be better from a pure *inference* perspective.
Since inference speed of SSMs is entirely governed by the state dimension, if one wants to maximize performance for a target inference efficiency (i.e. for a particular state size $\mathtt{N}$), then the increased expressivity of Mamba-1 might be better.
We haven't fully analyzed the (theoretical or empirical) tradeoffs here, and think this would be a cool direction for the community to dig in more!

### SSD vs. Attention

Compared, to standard (self-)attention, SSD also only has two differences:
1. The softmax normalization is dropped.
2. A separate elementwise mask matrix is applied multiplicatively.

The first difference can be interpreted as what reduces the effective state size of the model from infinite to finite, and improves its efficiency from quadratic to linear.

The second difference is what distinguishes SSD from standard linear attention.
One way to think of the mask is as **input-dependent relative positional encodings**.
Because of the mask $L$ in \eqref{eq:ssd-attention}, the standard attention score $Q_i K_j$ is attenuated by a weight $a_{i:j}^\times = a_i \cdots a_{j+1}$ which can be interpreted as a "discount factor" based on how far apart the positions $i$ and $j$ are.<d-footnote>This interpretation was concurrently espoused by Tobias Katsch's [GateLoop](https://arxiv.org/abs/2311.01927) paper<d-cite key="katsch2023gateloop"></d-cite></d-footnote>.
In its attention form, this input-dependent positional mask can be interpreted as the key factor that encodes the "selectivity" of Mamba!


## Best of Both Worlds

So why do we care that there are two views of this model?
Well, first of all, it's extremely mathematically interesting, as we'll cover in [Part 2], and we hope will inspire future directions.
But there are immediate practical benefits too! 

### Efficiency: the SSM and Attention Modes

The SSM \eqref{eq:ssm} and attention \eqref{eq:ssd-attention} modes represent two different ways of computing the same function,
so let's contrast them.

First, remember that one main reason why SSMs are interesting to begin with is because computing \eqref{eq:ssm} as a recurrence requires maintaining a *constant-size state* (size $\mathtt{N}$) and scales *linearly in the sequence length* $\mathtt{T}$.
The downside is that the raw FLOPs don't reflect actual speed in practice because of hardware considerations...

On the other hand, computing this sequence transformation $y = Mx$ through equation \eqref{eq:ssd-attention} takes quadratic time in the sequence length,
because we're materializing this $\mathtt{T} \times \mathtt{T}$ matrix.
But it can be fast in practice because it only uses matrix multiplications, which are extremely optimized on GPUs and TPUs.

### Efficiency: the SSD Mode

So if there are two equivalent ways of computing the same model, when should we use one mode or the other?
During inference, there's no trade-off: the SSM mode is designed for fast autoregressive inference.
But what about training?
There's a tension between FLOPs and hardware efficiency where the attention mode uses more FLOPs, but uses them more efficiently through matrix multiplications.

{% include figure.liquid loading="eager" path="assets/img/2024-05-31-mamba-2/ssd_algorithm.png" %}

It turns out we can get the best of both worlds by combining the algorithms!
There are two equivalent interpretations of this "state space dual" algorithm, either as
1. A block decomposition of a particular structured matrix that defines the SSD sequence transformation.
2. A "chunkwise" algorithm that splits the sequence into chunks, computes the quadratic attention form on each chunk, and adjusts the result by passing the SSM states between chunks.

We'll leave the details of this algorithm to the [full paper] (Section 6), as it requires a bit of machinery from the theory to derive.
But we do emphasize that the implementation of this algorithm isn't too complicated -- only ~30 lines of PyTorch,
which we provide in the paper (Listing 1) as well as in the [code release]!

The benefits of the SSD algorithm is that it preserves the same efficient FLOP counts as SSMs (compared to quadratic attention),
and also dramatically speeds up training comparing to general state space models.

|                        | Attention                | SSM             | SSD                |
| -------------          | -----------              | ----            | ---                |
| State size             | $\mathrm{T}$             | $\mathbf{N}$    | $\mathbf{N}$       |
| Training FLOPs         | $\mathrm{T}^2\mathrm{N}$ | $\mathbf{TN^2}$ | $\mathbf{TN^2}$    |
| Inference FLOPs        | $\mathrm{T}\mathrm{N}$   | $\mathbf{N^2}$  | $\mathbf{N^2}$     |
| (Naive) memory         | $\mathrm{T}^2$           | $\mathrm{TN}^2$ | $\mathbf{TN}$      |
| Matrix multiplication? | :heavy_check_mark:       | :x:             | :heavy_check_mark: |

[//]: # #### SSD Algorithm View 1: Matrix Decompositions
[//]: # 
[//]: # The way that we derived this algorithm is based on the SSD theory, which says that the state space model $Y = \mathsf{SSM}(A, B, C)(X)$ can actually be written as $Y = MX$ for a particular structured matrix $M$ called a **semiseparable matrix**.
[//]: # Then the question of computing the model efficiently is reduced to the problem of making this matrix multiplication as fast as possible.
[//]: # By decomposing it into blocks, we can leverage properties of semiseparable matrices to simplify the off-diagonal blocks.
[//]: # 
[//]: # #### SSD Algorithm View 2: Chunking and State Passing

## The Mamba-2 Architecture

Although the core contribution of Mamba-2 is the new SSD layer,
we also make some small changes to Mamba's neural network architecture.

{% include figure.liquid loading="eager" path="assets/img/2024-05-31-mamba-2/architecture_2.png" %}

The main change is producing the $(A, B, C)$ SSM parameters in parallel with the $X$ input, instead of sequentially.
This is partly motivated by the connections to attention;
but more pragmatically, it's simpler and more amenable to scaling techniques such as tensor parallelism, which will be covered in the next part of this series!

There are some other small differences which are covered in more detail in the paper.
However, we do want to emphasize that these architectural changes aren't really the main point of the model.

### Language Modeling

In terms of empirical results, we didn't test Mamba-2 as extensively as Mamba-1, but believe it should generally be on par or better.
Our full language model results use the same protocol as Mamba, and found slightly better scaling both at Chinchilla laws :warning: (figure).

{% include figure.liquid loading="eager" path="assets/img/2024-05-31-mamba-2/pile_8k_mamba2.png" %}

Fully trained models on the Pile dataset :warning: and the standard zero-shot downstream evaluations show similar trends.
We emphasize that even when the performance is comparable, Mamba-2 is *much* faster to train than Mamba-1!

### Synthetic Language Modeling: MQAR

More interestingly, we highlight the one synthetic task we tried.
Since the original Mamba paper, which investigated synthetics such as Synthetic Copying and Induction Heads,
many follow-up works have begun investigating harder associative recall tasks.
The multi-query associative recall (MQAR) task introduced by the Zoology and Based :warning: line of work
has become a de facto standard.

{% include figure.liquid loading="eager" path="assets/img/2024-05-31-mamba-2/mqar.png" %}

We ran a version of this task that's much harder than the one usually reported in the literature,
and found that Mamba-2 is substantially better than Mamba-1.
One reason for the improved performance is the much larger state size (up to $16\times$ larger than Mamba-1 here),
which was one of the primary motivations of Mamba-2 in the first place.

Interestingly, Mamba-2 also appears to be noticeably better than Mamba-1 on this particular task even when the state size is controlled.
We're not quite sure why to be honest, and it would be great to ablate the other aspects of the architecture to investigate...

## What's Next?

[AG: perhaps include a section to shout out related work and directions?]

---

# State Space Duality (Mamba-2) Part 2 - The Theory

In [Part 1] of this series, we defined the state space dual (SSD) *model*.
In isolation, this model is relatively simple to define,
and we claimed that it can be computed either as an SSM recurrence or with an attention-like pattern.
If you just want to use the model, feel free to stop here!

In this post, we'll dive into the theory behind the model.
We'll derive the SSD "duality" in two completely separate ways, one starting from the SSM perspective and one from the attention perspective.
Each method is actually much more broad than the SSD model itself,
and the union of these two strong generalizations is what we call the SSD *framework*.
This framework provides a rich body of connections between state space models, attention, and structured matrices.
While the SSD model can be viewed as a specific instantiation of each prong of the framework,
the SSD framework is much more general opens up many directions for future work.

#### The State Space Duality framework

{% include figure.liquid loading="eager" path="assets/img/2024-05-31-mamba-2/ssd_venn.png" title="Structured State Space Duality" caption="SSD Framework (red, blue): State space models (i.e. semiseparable matrices) and structured masked attention encapsulate large classes of efficient sequence models. Their intersection is the SSD model (purple)." %}


For each of the two parts of this framework, we'll
1. Define the general concepts
2. Show how the SSD model is an instantiation, and prove the duality
3. Suggest future directions for how the framework can be used

Note that this theory is *not necessary* to use the SSD model itself; this part of the series can be safely skipped for the practitioner that just wants to use SSD (Mamba-2).

## SSD Framework 1 -- Structured Matrix Transformations

The first framing of the duality will be from an SSM-centric perspective, where we'll prove the duality through the framework of **matrix sequence transformations** or "matrix mixers".

### Matrix Transformations
The idea is that many sequence models, i.e. sequence transformations $X \in \mathbb{R}^\mathtt{(T,P)} \mapsto Y \in \mathbb{R}^\mathtt{(T,P)}$,
can be written in the form of a single matrix multiplication $Y = M(X) \cdot X$ where $M$ is a matrix which can itself depend on $X$.
We call this a matrix sequence transformation, or matrix transformation for short.
In the literature sequence transformations have also been referred to as "sequence mixers", and matrix sequence transformations as matrix mixers.
There are many examples of these, which are distinguished by the structure of the $M$ matrix.
The de facto example is self-attention itself, where $M = \mathsf{softmax}(QK^\top)$ is the attention matrix.
Other examples include MLP-Mixer<d-cite key="tolstikhin2021mlp"></d-cite>,
FNet<d-cite key="lee2021fnet"></d-cite>,
and Monarch Mixer<d-cite key="dao2022monarch"></d-cite><d-cite key="fu2024monarch"></d-cite>,

Why do we care about these types of models?
> Writing a sequence model as a matrix transformation provides a powerful tool to understand the structure and characteristics of the model.

And although general non-linear RNNs such as LSTMs *cannot* be written as matrix mixers,
state space models can!
In fact, this is pretty easy to see by just unrolling the definition of the SSM recurrence.
The upshot is that the SSM \eqref{eq:ssm-transformation} can be written as a matrix transformation

$$
Y = \mathsf{SSM}(A, B, C)(X) = MX
$$

where $M_{ij} = 0$ for $i < j$ (i.e. it's lower triangular)
and otherwise
\begin{equation}
\label{eq:semiseparable}
M_{ij} = C_i^\top A_{i:j}^\times B_j := C_i^\top A_i \dots A_{j+1} B_j
\end{equation}

[//]: # $$
[//]: # M_{ij} =
[//]: # \begin{cases}
[//]: #   C_i^\top A_{i:j}^\times B_j := C_i^\top A_i \dots A_{j+1} B_j & i \ge j \\
[//]: #   0 & i < j
[//]: # \end{cases}
[//]: # $$

Drawing it out, this matrix looks like

$$
  \begin{bmatrix}
    C_0^\top B_0 & \\
    C_1^\top A_1 B_0 & C_1^\top B_1 & \\
    C_2^\top A_2A_1 B_0 & C_2^\top A_2 B_1 & C_2^\top B_2 \\
    \vdots & \vdots & \ddots & \ddots \\
    C_\mathtt{T}^\top A_{\mathtt{T}-1}\dots A_1 B_0 & C_\mathtt{T}^\top A_{\mathtt{T}-1}\dots A_2 B_1 & \dots & C_\mathtt{T}^\top A_{\mathtt{T}-1} B_{\mathtt{T}-2} & C_\mathtt{T}^\top B_{\mathtt{T}-1} \\
  \end{bmatrix}
$$

$$
(\text{Matrix Transformation Representation of State Space Models})
$$

### Semiseparable Matrices

This type of matrix in fact has a name: it's called a (triangular) **semiseparable matrix**,
and has been studied fairly extensively in other fields of engineering and computational linear algebra<d-cite key="vandebril2005bibliography"></d-cite>.
These matrices are (IMO) quite fundamental and beautiful,
and the full paper talks about more of their properties.
For example, an alternative characterization of semiseparable matrices is their *structured rank property*,
which says that every submatrix contained in the lower-triangular portion is low rank!

{% include figure.liquid loading="eager" path="assets/img/2024-05-31-mamba-2/semiseparable.png" %}

[//]: # The power of writing state space models as matrix transformations is that 

For our purposes, we'll care about this form mainly for the algorithmic considerations.
One of the central messages of this SSD paper is that:

> All algorithms for computing state space models can be viewed as structured matrix multiplication algorithms on semiseparable matrices.
{: .block-tip}

Let's see an easy instantiation of this, focusing on our main objective!

### Deriving the SSD Model's Duality (SSM →  Attention)

To show that equation \eqref{eq:ssd-attention} follows from equation \eqref{eq:ssm} (in the case of the SSD model, i.e. scalar SSM), we directly use the matrix form of the state space model \eqref{eq:semiseparable}.
Because the $A_t$ are all scalars in this case, they can be factored out of the entries

$$
C_i^\top A_{i:j}^\times B_j = A_{i:j}^\times \cdot (C_i^\top B_j)
$$

which directly implies equation \eqref{eq:semiseparable}.

In summary:

> The duality for the SSD model can be seen as two **different matrix multiplication algorithms** on the semiseparable matrix.
{: .block-tip}

- The linear form is a *structured matrix multiplication algorithm* that computes the outputs $Y_0, Y_1, \dots$ sequentially, leveraging the structure of the semiseparable matrix.
- The quadratic form is the *naive matrix multiplication algorithm* that materializes the quadratic matrix.

### Going Beyond SSD

The power of the semiseparable matrix representation applies to *all* state space models,
with various downstream implications.

#### Algorithms

Algorithmically, the Mamba-2 paper explores several consequences, such as:
1. The above duality result for the SSD model, i.e. a scalar-identity structured SSM.
2. New asymptotic efficiency results for state space models ([Theorem 3.7]), which follow directly from applying known results from the semiseparable matrix literature <d-cite key="pernet2016computing"></d-cite><d-cite key="pernet2018time"></d-cite><d-cite key="pernet2023exact"></d-cite>.
3. A more general hybrid algorithm that can be viewed as combining both the linear and quadratic forms to get the best of both worlds. This can be derived as a new matrix multiplication algorithm utilizing *block decompositions* of the semiseparable matrix. This is the subject of Part 3 of this blog post!

#### Understanding
Conceptually, the matrix transformation viewpoint helps provide a unifying view of sequence models.
Some example downstream ideas include:
- New sequence models: Restricting ourselves to matrix transformations reduces the problem of generating new sequence models to that of finding structured matrix classes with target properties. In ongoing work by my students, we study this point of view, and use it to derive the most natural bidirectional extension of Mamba (coming very soon!).
- Expressivity: Looking at the matrix transformation representation can help us understand what different models can represent from a linear algebraic perspective. In another ongoing work, we use this as a tool to study which subquadratic models are the most amenable to being distilled from Transformers.
- Interpretability: A concurrent work <d-cite key="ali2024hidden"></d-cite> derived the matrix formulation of SSMs and use it to probe the internal representations of Mamba models.

We're excited to see what algorithmic and conceptual ideas from the structured matrix literature can be applied to further improve state space models!



## SSD Framework 2 -- Structured Attention

The second framing of the duality is from an attention-centric perspective, where we'll prove the duality through the framework of **tensor contractions**.

Note that this is entirely independent of the previous [[matrix transformation viewpoint](#ssd-framework-1-structured-matrix-transformations)].

### Warm-up: Kernel Attention

For our purposes, we'll define attention as a function

$$
(Q^\mathtt{(T,N)}, K^\mathtt{(S,N)} , V^\mathtt{(S,P)} ) \mapsto Y^\mathtt{(T,P)}
$$

given by the pairwise matrix multiplications

$$
Y = (QK^\top) \cdot V
$$

{% details On Dimensions %}
Think of $\mathtt{P} = \mathtt{N}$ as the head dimension; technically speaking, in attention the $V$ head dimension $\mathtt{P}$ can differ from the $QK$ head dimension $\mathtt{N}$.
Think of $\mathtt{T}$ as the *target* sequence dimension and $\mathtt{S}$ as the *source* sequence dimension.
Giving these two axes different names will make the math more clear and also covers more general forms of attention such as cross-attention, where the source and target are separate sequences with different lengths.
However, for our purposes we'll assume the self-attention setting where $\mathtt{S}=\mathtt{T}$.
{% enddetails %}


{% details Why can we assume this form? %}
The usual form of attention $Y = f(QK^\top) \cdot V$ (e.g. where $f$ is the softmax function)
can, for essentially all functions $f$<d-footnote>And up to some additional massaging such as row-wise normalization, which is easy to handle</d-footnote>,
be written as $Y = \psi(Q)\psi(K)^\top \cdot V$ for some appropriate feature map $\psi$ (which may be infinite dimensional).
In this case, we can simply redefine $Q \leftarrow \psi(Q)$ and define $\mathtt{N}$ to be the **feature dimension** of the attention kernel to begin with.
Softmax attention, for example, can be represented with a particular infinite-dimensional feature map ($\mathtt{N}=\infty$) which represents the exponential kernel.
{% enddetails %}
We'll restrict ourselves to the case when $\psi$ is finite, which is sometimes called **kernel attention**.
Many, many variants have been proposed before!<d-cite key="katharopoulos2020transformers"></d-cite><d-cite key="peng2021random"></d-cite><d-cite key="choromanski2021rethinking"></d-cite><d-cite key="qin2022cosformer"></d-cite><d-cite key="zheng2022linear"></d-cite><d-cite key="wang2020linformer"></d-cite><d-cite key="xiong2021nystromformer"></d-cite>

Why do we care about this formulation?
When the sequence length $\mathtt{T}$ grows and the feature dimension $\mathtt{N}$ is small---commonly, in the regime when $\psi$ is simple such as an elementwise transform and so $\mathtt{N}$ is constant---then the cost of attention can be reduced from quadratic in $\mathtt{T}$ to linear.
This follows from simply computing the matrix multiplications in a different order

$$
Y = Q \cdot (K^\top V)
$$

This is a somewhat "folklore" interpretation of linear attention.<d-footnote>At least, one lineage of efficient attention; other varieties exist, such as those based on sparsity or hashing. We reserve the term "linear attention" to those related to Katharopoulos et al.<d-cite key="katharopoulos2020transformers"></d-cite>, or more broadly low-rank attention.</d-footnote>

> The most common way of linearizing attention is usually viewed as a consequence of the *associativity of matrix multiplication*

### (Causal) Linear Attention

However, once the basic kernel attention is slightly modified, we can no longer use the associativity of matrix multiplication directly.

The seminal **Linear Attention (LA)** framework of Katharopoulos et al. <d-cite key="katharopoulos2020transformers"></d-cite> shows that it can still be extended to the important case of incorporating causality into attention, for autoregressive settings such as language modeling.

Let's be a lot more explicit about how it works.
The quadratic form of **causal linear attention** is
\begin{equation}
\label{eq:quadratic-kernel-attention}
Y = (L \circ QK^\top) \cdot V
\end{equation}
where

$$
L =
\begin{bmatrix} 1 \\ \vdots & \ddots \\ 1 & \dots & 1 \end{bmatrix}
$$

is the **causal mask** matrix.

The issue is: once the $L$ mask is incorporated into \eqref{eq:quadratic-kernel-attention}, we can no longer directly apply matrix associativity!
This is the problem that the original Linear Attention paper addresses.
What they show is that \eqref{eq:quadratic-kernel-attention} is equivalent to a different form
which avoids materializing the quadratic $QK^\top$ attention matrix and has linear time complexity

$$
Y = Q \cdot \mathsf{cumsum}(K^\top V)
$$

As far as we're aware this wasn't explicitly proved in the paper, although it isn't too hard to write out the summation to show it.

What we'll do is prove this equivalence in essentially one line, while revealing *exactly* where the "linear" part of Linear Attention comes from, and how to strongly generalize it.

Spoiler alert:
> #### Where does the cumsum in Linear Attention come from?
>
> The appearance of the *cumulative sum* in linear attention is exactly equivalent to the fact that the causal mask $L$, as a matrix multiplication, encodes cumulative sums:
>
> $$y = L \cdot x \iff y = \mathsf{cumsum}(x)$$
{: .block-tip }

### A Tensor Contraction Proof of Linear Attention

Let's write out the quadratic form of linear attention \eqref{eq:quadratic-kernel-attention}
very explicitly in **tensor contraction** or [einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) notation, with shape annotations:


[//]: #    Q &= \mathsf{input} && \qquad \mathtt{(T,N)} \\
[//]: #    K &= \mathsf{input} && \qquad \mathtt{(S,N)} \\
[//]: #    V &= \mathsf{input} && \qquad \mathtt{(S,P)} \\

$$
  \begin{aligned}
    G &= \mathsf{contract}(\mathtt{TN, SN} \to \mathtt{TS})(Q, K) \\
    M &= \mathsf{contract}(\mathtt{TS, TS} \to \mathtt{TS})(G, L) \\
    Y &= \mathsf{contract}(\mathtt{TS, SP} \to \mathtt{TP})(M, V) 
  \end{aligned}
$$

\begin{equation}
\label{eq:sma-quad}
(\text{Structured Masked Attention - Quadratic Form})
\end{equation}


With this notation, we can notice that this sequence of contractions can be written as a *single four-way contraction*

\begin{equation}
  \label{eq:sma}
  y = \mathsf{contract}(\mathtt{TN},\mathtt{SN},\mathtt{SP},\mathtt{TS} \to \mathtt{TP})(Q, K, V, L)
  .
\end{equation}

And finally, it can be computed with any other contraction ordering. In particular,
we can perform pairwise reductions on the order $V, K, L, Q$ instead of $Q, K, L, V$

$$
  \begin{aligned}
    Z &= \mathsf{contract}(\mathtt{SP},\mathtt{SN} \to \mathtt{SPN})(V, K)  \\
    H &= \mathsf{contract}(\mathtt{TS},\mathtt{SPN} \to \mathtt{TPN})(L, Z) \\
    Y &= \mathsf{contract}(\mathtt{TN},\mathtt{TPN} \to \mathtt{TP})(Q, H)
  \end{aligned}
$$

\begin{equation}
\label{eq:sma-lin}
(\text{Structured Masked Attention - Linear Form})
\end{equation}

Now the key observation is that the second line of \eqref{eq:sma-lin} is simply a matrix multiplication by $L$,
which can be computed with a cumulative sum.

That's the entire proof of linear attention! The beauty of it is that we didn't have to write out a single summation, which was abstracted out into a tensor contraction.
In particular, the second contraction in equation \eqref{eq:sma-lin} is simply a matrix multiplication by the mask matrix $L$.

This immediately proves our claim about the [cumsum in linear attention](#where-does-the-cumsum-in-linear-attention-come-from).
Moreover, this immediately reveals that the efficiency of linear attention can be made *much more general*...


### Structured Masked Attention

The critical observation is that in order for 
\eqref{eq:sma-lin} to be fast,
all that is necessary is for $L$ to be *any structured matrix* --
in other words any matrix that has subquadratic matrix-vector multiplication.

This immediately motivates one of the main prongs of the SSD framework,
which can be seen as a strong generation of LA.

> #### Definition: Structured Masked Attention
>
> **Structured masked attention (SMA)** is defined as the *four-way tensor contraction* \eqref{eq:sma} using an attention mask $L$ that is a structured matrix.  
{: .block-tip }

> SMA has **dual quadratic and linear**<d-footnote>Assuming that the structured matrix $L$ has linear time matrix-vector multiplication</d-footnote> **modes** which are simply *two different pairwise reduction orders* \eqref{eq:sma-quad} and \eqref{eq:sma-lin}.
{: .block-tip }


Finally, let's just connect this back to the commonly held view of linear attention as matrix multiplication associativity.

> Although it is commonly believed that incorporating attention masks $L$ prevents matrix multiplication reordering, it turns out to still be compatible.
> In particular, **associativity of matrix multiplication** is a special case of **tensor contraction reduction orders**;
> although the former no longer applies, the latter can integrate the attention mask $L$.


Next, let's look at some consequences of the structured attention framework.

### Deriving the SSD Model's Duality (Attention →  SSM)

Recall that the SSD model is defined as either a scalar-identity SSM in equation \eqref{eq:ssm},
or through the attention-like form in equation \eqref{eq:ssd-attention}.

To show the equivalence of these forms, we simply recognize that \eqref{eq:ssd-attention} is a special case of structured masked attention where the mask matrix is

$$
  L =
  \begin{bmatrix}
    1 & \\
    a_1 & 1 & \\
    a_2a_1 & a_2 & 1 \\
    \vdots & \vdots & \ddots & \ddots \\
    a_{\mathtt{T}-1}\dots a_1 & a_{\mathtt{T}-1}\dots a_2 & \dots & a_{\mathtt{T}-1} & 1 \\
  \end{bmatrix}
  .
$$

\begin{equation}
\label{eq:1-ss}
(\text{1-semiseparable (1-SS) matrix})
\end{equation}

We call this a **1-semiseparable (1-SS) matrix**, for reasons that are explained in more detail in the Mamba-2 paper.

Thus, we can also say that the SSD model is **1-semiseparable masked attention** or **1-SS SMA**.

To prove that this can be written as an SSM, we simply appeal to the SMA framework, which says that the dual form of this model can be computed through matrix multiplication by $L$.
So how fast is that?
It's not too hard to see that multiplication $y = Lx$ can be computed in linear time through a scalar recurrence:

$$
\begin{aligned}
y_0 &= x_0 \\
y_1 &= a_1 x_0 + a_1 \\
y_2 &= a_2a_1 x_0 + a_2 x_1 + x_2 = a_2 y_1 + x_2 \\
\vdots & \qquad \vdots
\end{aligned}
$$

This corresponds exactly to the original SSM recurrence!

(In fact, multiplication by 1-SS matrices $L$ can be computed in a *lot* more ways, which we compile in the full paper! Alternative algorithms can reveal more insights: for example, the associative scan algorithm used by S5 <d-cite key="smith2023s5"></d-cite> and Mamba can also be shown to be a structured matrix multiplication algorithm on 1-SS matrices.)


### Going Beyond SSD

Structured masked attention not only helps define the SSD model and prove its duality,
but it is a much broader framework of efficient attention models.

{% include figure.liquid loading="eager" path="assets/img/2024-05-31-mamba-2/sma.png" %}

Prior examples include the original linear attention as well as the recent Retentive Network (RetNet) model<d-cite key="sun2023retentive"></d-cite>.
These can be viewed as direct special cases of SSD.
But beyond SSD, we can define classes of efficient attention by replacing the mask $L$ with *any structured matrix*.
As a suggestion, we think that Toeplitz or Fourier structured attention may be interesting to consider because they might encode different forms of positional information.

Additionally, other forms of structure can be incorporated into the $L$ mask.
For example, another extension my students are developing is viewing SSD (and recurrences in general) as an algorithm operating on *directed line graphs*,
and generalizing it to incorporate arbitrary graph structures.


## State Space Duality (Redux)

We'll end this post with a brief recap of what we've covered.

The **SSD framework** consists of the two broad approaches covered in this post, which is summarized by the [[main figure](#the-state-space-duality-framework)]:
1. Viewing state space models through [[structured matrix transformations](#ssd-framework-1-structured-matrix-transformations)]
2. Viewing linear attention through [[structured masked attention](#ssd-framework-2-structured-attention)]

The [[SSD model](#the-state-space-dual-model)] is a particular model which is the purple intersection in the figure, which can be viewed as an instance of either part of the SSD framework, and in particular has dual quadratic and linear forms that can be derived from either representation.


| SSD Framework                                       | Structured Matrix Transformations                                  | Structured Attention                                              |
| -------------                                       | -----------                                                        | ----                                                              |
| Which sequence models does this focus on?           | State space models (SSM)                                           | (Linear) Attention                                                         |
| The SSD model is an instantiation as...             | Scalar state space models <br> ($A_t$ is a scalar-identity matrix) | 1-semiseparable masked attention <br> ($L$ mask is a 1-SS matrix) |
| The linear-quadratic duality is revealed through... | Matrix multiplication algorithms                                   | Tensor contraction reduction orderings                            |

## Code

{% highlight python linenos %}

def test():
  return None

{% endhighlight %}

{% highlight python %}

def segsum(x):
    """Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,
       which is equivalent to a scalar SSM."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd(X, A, B, C, block_len=64, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state

{% endhighlight %}

Backticks:

```javascript
def segsum(x):
    """Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,
       which is equivalent to a scalar SSM."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd(X, A, B, C, block_len=64, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state
```

`<d-code>`:

<d-code block language="javascript">
def segsum(x):
    """Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,
       which is equivalent to a scalar SSM."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd(X, A, B, C, block_len=64, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state
</d-code>
