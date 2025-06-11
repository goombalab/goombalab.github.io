---
layout: distill
title: "Understanding and Improving Length Generalization in Recurrent Models"
description: 
tags:
giscus_comments: false
date: 2025-06-09
featured: false
# thumbnail: assets/img/2024-07-16-hydra/sequence_mixer_trans.png


authors:
  - name: Ricardo Buitrago Ruiz
    url:
    affiliations:
      name: CMU, Cartesia AI
  - name: Albert Gu
    url:
    affiliations:
      name: CMU, Cartesia AI

bibliography: ricardo.bib

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
  - name: Current recurrent models fail to deliver on their promise - the length generalization problem
  - name: Why do recurrent models fail to length generalize?

---

[[Paper LINK MISSING](XXX)]


## Current recurrent models fail to deliver on their promise - the length generalization problem

Transformers <d-cite key="attention_is_all_you_need"></d-cite> have become the most popular architecture in the Machine Learning community thanks to their strong performance across many tasks spanning from lanugage modeling to image generation. However, they suffer from two fundamental bottlenecks: (1) they have quadratic complexity over the sequence length, which make it extremely hard to process long sequences; and (2) they do not naturally have a concept of order elements in the sequence, so they rely on positional embeddings to treat past elements differently from recent elements in the sequence. As an attempt to overcome these two limitations, several linear <em>recurrent models</em> have been proposed, such as Mamba <d-cite key="mamba"></d-cite><d-cite key="mamb2a"></d-cite>, which apply a recurrence to the sequence. Thus, <strong>in theory</strong> they can (1)  <strong>efficiently process long sequences </strong>, since recurrences have linear complexity over the sequence length; and (2)  <strong>naturally process sequences of any length </strong> by simply rolling out the state recurrence (i.e. they do not have positional embeddings). Thus, the prevailing attitude is that even if Transformers were strictly more expressive in short sequences, we fundamentally should tend towards linear recurrent arcthiectures for their efficient processing of long sequences.

However,  <strong>in practice many modern recurrent architectures have extremely low performance when processing long sequences </strong> (specifically, when processing sequences longer than what they have been trained on). In the following figure, we show the performance of the official Mamba-2 checkpoints <d-cite key="mamb2a"></d-cite> as a function of the sequence position $t$ (using perplexity, the lower the better). It can be seen that for positions $t$ beyond the training context $T=2048$, these models become virtually useless: they fail to <em>length generalize</em>.

{% include figure.liquid loading="eager" path="assets/img/2025-06-11-length-generalization/mamba2-poswise.png" %}

We have a paradox here: modern recurrent models are failing to deliver on their promise of enabling the processing long sequences (which was their original motivation!) However, in this blog post we will show that <strong>length generalization is easily achievable in many recurrent models through simple training interventions</strong>, thus this is more an unrealised potential rather than a fundamental limitation of recurrent models. 

## Why do recurrent models fail to length generalize?

To output the element at position $t+1$, recurrent models apply an operation on the previous element $x_t$ and a (fixed size) <it>recurrent state</it> $h_t$ (i.e. compressed information from all previous elements of the sequence from $0$ to $t+1$), so we can write $x_{t+1} = f(x_t, h_t)$ for some function $f$. The function $f$ does not depend on the position, so in theory recurrent models should naturally be able to process any sequence length.

However, in our work we show that <strong>the distribution of the state $h_t$ changes over time</strong>, thus even if $f$ might work properly up to some $T$, other $h_t$ with $T>t$ might be significantly different, and thus $f$ fails to produce the correct output. Indeed, in the following figure we show how the norm of the state of Mamba-2 <d-cite key="mamb2a"></d-cite> changes significantly over time:

{% include figure.liquid loading="eager" path="assets/img/2025-06-11-length-generalization/statemetrics_full.png" %}

At first, one might think that as the sequence position increases, the fixed-size state needs to remember information from a longer sequence and thus somehow saturates. However, in this work we show that this intuition is not correct. Indeed, if this was the case the recurrent model would struggle to "remember" elements in the sequence that are far away. In our work, we introduce Effective Remembrace to measure how much an autoregressive is effectively remembering previous tokens. Denote by $q(\cdot | \text{context})$ the probabilities that an autoregressive sequential model outputs for an element given a context. Then, we define: 
$$
    \text{EffRem}_T(t) = d(q(\cdot |x_{0:T}),q(\cdot| x_{t:T}))
$$

Where $d$ is a distance between probability distributions (total variation). If $\text{EffRem}_T(t)=0$, this means that the predictions using $x_{t:T}$ and using $x_{0:T}$ are the same, meaning that the model does not ``effectively remember'' any of the past tokens $x_{0:t-1}$. Conversely, if $\text{EffRem}_T(t)$ is high, the model is substantially influenced by the tokens $x_{0:t-1}$, since removing them from the context changes the prediction significantly.

It turns out that models that fail to length generalize have very high $\text{EffRem}_T(t)$ for small $t$, meaning that the models are disproportionately impacted by early elements of the sequence.

{% include figure.liquid loading="eager" path="assets/img/2025-06-11-length-generalization/mamba2-effrem.png" %}


Thus, it is not that the state cannot remember all information from the sequence; rather, in a sense it is so expressive that early elements can completely change its prediction (which is not desirable, as the prediction should mostly focus on the recent context). This indicates that the failure to length generalize is not due to lack of capacity, rather due to an undesired behavior of a state that is far too expressive. This insight made us think of this issue as an overfitting problem, which should be solved with training interventions rather than with architecture modifications.