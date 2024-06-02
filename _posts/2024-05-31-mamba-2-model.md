---
layout: distill
title: State Space Duality (Mamba-2) Part I - The Model
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

