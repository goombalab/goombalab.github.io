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
  - name: The unexplored states hypothesis

---

[[Paper LINK MISSING](XXX)]


## Current recurrent models fail to deliver on their promise - the length generalization problem

Transformers <d-cite key="attention_is_all_you_need"></d-cite> have become the most popular architecture in the Machine Learning community thanks to their strong performance across many tasks spanning from lanugage modeling to image generation. However, they suffer from two fundamental bottlenecks: (1) they have quadratic complexity over the sequence length, which make it extremely hard to process long sequences; and (2) they do not naturally have a concept of order elements in the sequence, so they rely on positional embeddings to treat past elements differently from recent elements in the sequence. As an attempt to overcome these two limitations, several linear <em>recurrent models</em> have been proposed, such as Mamba <d-cite key="mamba"></d-cite><d-cite key="mamba2"></d-cite>, which apply a recurrence to the sequence. Thus, <strong>in theory</strong> they can (1)  <strong>efficiently process long sequences </strong>, since recurrences have linear complexity over the sequence length; and (2)  <strong>naturally process sequences of any length </strong> by simply rolling out the state recurrence (i.e. they do not have positional embeddings). Thus, the prevailing attitude is that even if Transformers were strictly more expressive in short sequences, we fundamentally should tend towards linear recurrent arcthiectures for their efficient processing of long sequences.

However,  <strong>in practice many modern recurrent architectures have extremely low performance when processing long sequences </strong> (specifically, when processing sequences longer than what they have been trained on). In the following figure, we show the performance of the official Mamba-2 checkpoints <d-cite key="mamba2"></d-cite> as a function of the sequence position $t$ (using perplexity, the lower the better). It can be seen that for positions $t$ beyond the training context $T=2048$, these models become virtually useless: they fail to <em>length generalize</em>.

{% include figure.liquid loading="eager" path="assets/img/2025-06-11-length-generalization/mamba2-poswise.png" %}

We have a paradox here: modern recurrent models are failing to deliver on their promise of enabling the processing long sequences (which was their original motivation!) However, in this blog post we will show that <strong>length generalization is easily achievable in many recurrent models through simple training interventions</strong>, thus this is more an unrealised potential rather than a fundamental limitation of recurrent models. 

## Why do recurrent models fail to length generalize?

To output the element at position $t+1$, recurrent models apply an operation on the previous element $x_t$ and a (fixed size) <it>recurrent state</it> $h_t$ (i.e. compressed information from all previous elements of the sequence from $0$ to $t+1$), so we can write $x_{t+1} = f(x_t, h_t)$ for some function $f$. The function $f$ does not depend on the position, so in theory recurrent models should naturally be able to process any sequence length.

However, in our work we show that <strong>the distribution of the state $h_t$ changes over time</strong>, thus even if $f$ might work properly up to some $T$, other $h_t$ with $T>t$ might be significantly different, and thus $f$ fails to produce the correct output. Indeed, in the following figure we show how the norm of the state of Mamba-2 <d-cite key="mamba2"></d-cite> changes significantly over time:

{% include figure.liquid loading="eager" path="assets/img/2025-06-11-length-generalization/statemetrics_full.png" %}

At first, one might think that as the sequence position increases, the fixed-size state needs to remember information from a longer sequence and thus somehow saturates. However, in this work we show that this intuition is not correct. Indeed, if this was the case the recurrent model would struggle to "remember" elements in the sequence that are far away. In our work, we introduce Effective Remembrace to measure how much an autoregressive is effectively remembering previous tokens. Denote by $q(\cdot \| \text{context})$ the probabilities that an autoregressive sequential model outputs for an element given a context. Then, we define $\text{EffRem}_T(t) = d\(q\(\cdot \| x\[0:T\],q(\cdot \| x\[t:T\]\)\)$, where $d$ is a distance between probability distributions (e.g. Total Variation). If $\text{EffRem}_T(t)=0$, this means that the predictions using $x\[t:T\]$ and using $x\[0:T\]$ are the same, meaning that the model does not ``effectively remember'' any of the past tokens $x\[0:t-1\]$. Conversely, if $\text{EffRem}_T(t)$ is high, the model is substantially influenced by the tokens $x\[0:t-1\]$, since removing them from the context changes the prediction significantly.

It turns out that models that fail to length generalize have very high $\text{EffRem}_T(t)$ for small $t$, meaning that the models are disproportionately impacted by early elements of the sequence.

{% include figure.liquid loading="eager" path="assets/img/2025-06-11-length-generalization/mamba2-effrem.png" %}


Thus, it is not that the state cannot remember all information from the sequence; rather, in a sense it is so expressive that early elements can completely change its prediction (which is not desirable, as the prediction should mostly focus on the recent context). This indicates that the failure to length generalize is not due to lack of capacity, rather due to an undesired behavior of a state that is far too expressive. This insight made us think of this issue as an overfitting problem, which should be solved with training interventions rather than with architecture modifications.

## The unexplored states hypothesis
The results for Effective Remembrance suggest that the models <em>can</em> remember information from long contexts, but the failure to length generalize indicate that they have not <em>learnt</em> to do so - i.e. the have not been trained on certain state distributions. Thus, we propose the <em><strong>unexplored states hypothesis</strong>: Recurrent models fail to length generalize when they are trained only on a subset of all attainable state distributions---i.e. on a subset of the states that would be attained if the state recurrence was rolled out indefinitely. When trained for long enough, the model overfits to this subset and performs poorly on long sequences because it encounters unexplored state distributions.</em>

## Interventions to enable length generalization
The unexplored states hypothesis indicates that length generalization can be achieved not by changing the architecture or its mechanisms, but by training the model on a more diverse set of state distributions - in particular, on the distributions that arise when rolling out the state recurrence on long sequences. To do so, we could directly train the model on longer sequences, but this might not always be possible due to GPU memory constraints or due to lack of sufficently long training sequences. Thus, in our work we explore interventions on the <em>initial state</em> of the recurrence. Most modern architectures assume a zero initial state in the recurrence, meaning that the first state update does not take into account any previous state. In our work, we study four interventions on the initial state, which are aimed at letting the model explore more state distributions. Note that using with a non-zero initial state is equivalent to starting to process the sequence with some previous context.

The four training interventions can be seen as sampling the initial state from four different distributions, that progressively get closer to the distribution of attainable states:
1. Random Noise: The state is initialized with an IID Gaussian with zero mean and a chose standard deviation.
2. Fitted Noise: During training, we record the mean and standard deviation of the final states of the sequences across all layers and heads. Then, we initialize the state with IID Gaussian noise with those fitted means and stadard deviations.
3. State Passing: We use the final state of a previous (unrelated) sequence as the initial state (note that this is equivalent to sampling from the distribution of attainable states at the training context $T$).
4. Truncated Backpropagation Through Time (TBTT) <d-cite key="TBTT_1990"></d-cite> <d-cite key="TBTT_sutskever"></d-cite>: In this case, we split a long sequence into smaller chunks, and use the final state of each chunk as the initial state of the next one. This is equivalente to processing the whole sequence, yet stopping the gradient propagation between chunks.

The following figure shows the results for each intervention:
{% include figure.liquid loading="eager" path="assets/img/2025-06-11-length-generalization/training_interventions.png" %}

Firstly, both State Passing and TBTT - which are the interventions that are closer to realistic states - allow length generalization in sequences order of magnitude longer than those seen during training, which is the first important takeaway of our work: <strong>length generalization is expected to be readily achievable in recurrent models through simple training interventions</strong>. Secondly, we can infer properties of the distribution of states of recurrent models looking at the performance of the Random Noise and Fitted Noise interventions. The Random Noise intervention fails to length generalize in the 370m, but the Fitted Noise intervention works, suggesting that the distribution of attainable states cannot be approximated with a Gaussian with fixed variance, but it can be approximated with an IID Gaussian with fitted variance in each layer and head of the state. However, the Fitted Noise intervention fails to achieve length generaliation in the 1.3b model, indicating that the state probably has complex dependency relationships and thus cannot be approximated with IID values.

## Performance on long context tasks tasks
These interventions enable length <em>generalization</em> (i.e. not having decreased peformance after the training context $T$), but it is reasonable to ask whether they actually enable modeling complex relationships between different parts of the sequence, like  elements that are separated by more than $T$ position. In our work we answer affirmatively by showing the results on thee long context tasks.

<strong>BABILong</strong><d-cite key="babilong"></d-cite>. BABILong is a challenging benchmark which tests both the common sense understanding of a model as well as its ability to capture long range dependencies in text. In the figure below it can be observed that State Passing enhances the length extrapolation capabilities of the model in both the few shot and finetuned settings (we recall that the model is trained and finetuned on sequences of length 2048). Therefore, State Passing is not only useful in fixing the diverging perplexity of established language models, but also in enhancing their ability to solve long context reasoning tasks.

{% include figure.liquid loading="eager" path="assets/img/2025-06-11-length-generalization/babilong.png" %}

<strong>Passkey retrieval</strong><d-cite key="landmark_attention_mohtashami2023randomaccess"></d-cite>. The passkey retrieval task requires the model to retrieve a 5-digit passkey inserted at a given depth of a long context. In the figure below we show that models finetuned with fitted noise are capable of handling relationships between tokens that are much more than 2k positions apart.


{% include figure.liquid loading="eager" path="assets/img/2025-06-11-length-generalization/passkey.png" %}

<strong>Synthetic Copying</strong><d-cite key="transformers-better-copying-pmlr-v235-jelassi24a"></d-cite>. The synthetic copying task  consists in copying an arbitrary sequence of tokens. In the table below we show that using State Passing during training greatly improves length generalization in sequences more than three times longer. Thus, State Passing helps the model solve long context tasks that are harder than those seen during training.

{% include figure.liquid loading="eager" path="assets/img/2025-06-11-length-generalization/synthetic_copying.png" %}






