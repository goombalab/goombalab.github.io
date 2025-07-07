---
layout: distill
title: "Understanding and Improving Length Generalization in Recurrent Models"
description: 
tags:
giscus_comments: false
date: 2025-07-06
featured: false
thumbnail: assets/img/2025-07-06-length-generalization/thumbnail4.png


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
  - name: Existing Recurrent Models Still Fall Short
  - name: Why Do Recurrent Models Fail to Length Generalize? The Unexplored States Hypothesis
  - name: Interventions to Enable Length Generalization
  - name: Performance on Long Context Tasks
  - name: A Deeper Look into How Recurrent Models Process Context
  - name: Conclusion

---

[[Paper](https://arxiv.org/abs/2507.02782)]


## Existing Recurrent Models Still Fall Short

<div style="text-align: justify; margin-bottom: 1em;">
Linear recurrent models such as Mamba <d-cite key="mamba"></d-cite><d-cite key="mamba2"></d-cite> and linear attention <d-cite key="LA_katharopoulos2020transformers"></d-cite><d-cite key="RQKV"></d-cite><d-cite key="gated_linear_attention_yang2024gla"></d-cite><d-cite key="delta_net"></d-cite> possess <strong>a remarkable feature: they can process extremely long sequences</strong>, which is key for applications that require long context reasoning (like summarizing long texts or agents with long term memory). Indeed, this is their key advantage over their main competitor, the Transformers <d-cite key="attention_is_all_you_need"></d-cite>, which are bottlenecked by their finite context window and quadratic complexity over the sequence length.
</div>

<div style="text-align: justify; margin-bottom: 1em;">
Previously, the issue with recurrent models was their performance: on short sequences they were less capable than Transformers. But recent architecture breakthroughs have improved the performance of recurrent models and brought them on par with Transformers, to the point that they are currently used in several industry applications like audio modeling <d-cite key="goel2024sonic"></d-cite> or code completion <d-cite key="mistral2024codestral"></d-cite>. However, several recent works have found out that <em>recurrent models still fall short</em>: they might have comparable performance to Transformers, <strong> but in many cases they struggle to generalize past the training length.</strong>
</div>


<!-- Transformers <d-cite key="attention_is_all_you_need"></d-cite> have become the most popular architecture in the Machine Learning community thanks to their strong performance across many tasks spanning from language modeling to image generation. However, they suffer from two fundamental bottlenecks: (1) they have quadratic complexity over the sequence length, which make it extremely hard to process long sequences; and (2) they do not naturally have a concept of order elements in the sequence, so they rely on positional embeddings to treat past elements differently from recent elements in the sequence. As an attempt to overcome these two limitations, several linear <em>recurrent models</em> have been proposed, such as Mamba <d-cite key="mamba"></d-cite><d-cite key="mamba2"></d-cite>, which apply a recurrence to the sequence. Thus, <strong>in theory</strong> they can (1)  <strong>efficiently process long sequences </strong>, since recurrences have linear complexity over the sequence length; and (2)  <strong>naturally process sequences of any length </strong> by simply rolling out the state recurrence (i.e. they do not have positional embeddings). Thus, the prevailing attitude is that even if Transformers were strictly more expressive in short sequences, we fundamentally should tend towards linear recurrent arcthiectures for their efficient processing of long sequences. -->

<div style="text-align: justify; margin-bottom: 1em;">
Indeed, we show the performance of the official Mamba-2 checkpoints <d-cite key="mamba2"></d-cite> as a function of the sequence position $t$ (using perplexity, the lower the better). It can be seen that for positions $t$ beyond the training context $T=2048$, these models become virtually useless: they fail to <em>length generalize</em>.
</div>


<div style="max-width: 500px; margin: 0 auto; text-align: center;">


{% include figure.liquid loading="eager" path="assets/img/2025-07-06-length-generalization/mamba2_poswise_reduced2.png" %}

</div>



<!-- We have a paradox here: modern recurrent models are failing to deliver on their promise of enabling the processing long sequences (which was their original motivation!) However, in this blog post we will show that <strong>length generalization is easily achievable in many recurrent models through simple training interventions</strong>, thus this is more an unrealised potential rather than a fundamental limitation of recurrent models.  -->
<div style="text-align: justify; margin-bottom: 1em;">
This is an issue: existing recurrent models have low performance on long sequences, and are not much more efficient than Transformers in shorter sequences; so they seem to be falling short on both sides.
</div>

<div style="text-align: justify; margin-bottom: 1em;">
Does this mean that recurrent models are useless? Not at all! In our work, we show that <strong>length generalization is easily achievable in recurrent models through simple training interventions: post-training for 500 steps (~0.1% of the pre-training budget) enables length generalization in up to 256k sequences!</strong> Therefore, recurrent models possess an <em>unrealised potential</em> rather than a <em>fundamental limitation</em>.
</div>


## Why Do Recurrent Models Fail to Length Generalize? The *Unexplored States Hypothesis* 

<div style="text-align: justify; margin-bottom: 1em;">
For an input sequence with $t$ elements $(x_1, x_2, ..., x_{t-1}, x_t)$, recurrent models compress the input context $(x_1, x_2, ..., x_{t-1})$ into a fixed-size <em>recurrent state</em> $h_{t-1}$. At time $t=0$, the state is initialized with some value $h_{-1}$, and then it is updated at each $t$ with an update function $f$:
</div>
<div style="text-align: center;">
$ h_t = f(h_{t-1}, x_t) $
</div>

<div style="text-align: justify; margin-bottom: 1em;">
Similarly, the output at time $t$ only depends on the state $h_t$ and the current input $x_t$, i.e. for some other function $g$ the output $y_t$ can be written as
</div>
<div style="text-align: center;">
 $y_t = g(h_t, x_t)$
 </div>

 <div style="text-align: justify; margin-bottom: 1em;">
 The functions $f$ and $g$ do not depend on the position $t$, so in theory recurrent models can naturally process any sequence length. But then, how can it be that they fail when $t$ is large?
</div>

<div style="text-align: justify; margin-bottom: 1em;">
In our work we show that <strong>the distribution of the state $h_t$ changes over time</strong>. Therefore, even if $g$ and $f$ work correctly up to some $T$, other $h_t$ with $t>T$ might be significantly different, and thus the model fails to produce the correct output. Indeed, in the following figure we show how the norm of the state of Mamba-2 <d-cite key="mamba2"></d-cite> increases significantly over time:
</div>
<div style="max-width: 400px; margin: 0 auto; text-align: center;">
{% include figure.liquid loading="eager" path="assets/img/2025-07-06-length-generalization/statemetrics_base.png" width="0.1" %}
</div>
<div style="text-align: justify; margin-bottom: 1em;">
This explains why recurrent models fail to length generalize: when processing sequences longer than those seen during training, they encounter states $h_t$ that have not been explored during training, and thus they have not learnt to process them. Based on this insight, we propose the <strong>unexplored states hypothesis</strong> to explain the failure to length generalize:
</div>

> #### Unexplored States Hypothesis
> <div style="text-align: justify; margin-bottom: 1em;">Recurrent models fail to length generalize when they are trained only on a <strong>subset of all attainable state distributions</strong>&mdash;i.e. on a subset of the states that would be attained if the state recurrence was rolled out indefinitely.  </div>
><div style="text-align: justify; margin-bottom: 1em;"> When trained for long enough, the models <strong>overfit to this subset</strong> and perform poorly on long sequences because they <strong>encounter unexplored state distributions</strong>. </div>

<!-- At first, one might think that as the sequence position increases, the fixed-size state needs to remember information from a longer sequence and thus somehow saturates. However, in this work we show that this intuition is not correct. Indeed, if this was the case the recurrent model would struggle to "remember" elements in the sequence that are far away. In our work, we introduce Effective Remembrace to measure how much an autoregressive is effectively remembering previous tokens. Denote by $q(\cdot \| \text{context})$ the probabilities that an autoregressive sequential model outputs for an element given a context. Then, we define $\text{EffRem}_T(t) = d\(q\(\cdot \| x\[0:T\],q(\cdot \| x\[t:T\]\)\)$, where $d$ is a distance between probability distributions (e.g. Total Variation). If $\text{EffRem}_T(t)=0$, this means that the predictions using $x\[t:T\]$ and using $x\[0:T\]$ are the same, meaning that the model does not ``effectively remember'' any of the past tokens $x\[0:t-1\]$. Conversely, if $\text{EffRem}_T(t)$ is high, the model is substantially influenced by the tokens $x\[0:t-1\]$, since removing them from the context changes the prediction significantly.

It turns out that models that fail to length generalize have very high $\text{EffRem}_T(t)$ for small $t$, meaning that the models are disproportionately impacted by early elements of the sequence. -->


<!-- Thus, it is not that the state cannot remember all information from the sequence; rather, in a sense it is so expressive that early elements can completely change its prediction (which is not desirable, as the prediction should mostly focus on the recent context). This indicates that the failure to length generalize is not due to lack of capacity, rather due to an undesired behavior of a state that is far too expressive. This insight made us think of this issue as an overfitting problem, which should be solved with training interventions rather than with architecture modifications. -->

<!-- ## The unexplored states hypothesis
The results for Effective Remembrance suggest that the models <em>can</em> remember information from long contexts, but the failure to length generalize indicate that they have not <em>learnt</em> to do so - i.e. the have not been trained on certain state distributions. Thus, we propose the <em><strong>unexplored states hypothesis</strong>: Recurrent models fail to length generalize when they are trained only on a subset of all attainable state distributions---i.e. on a subset of the states that would be attained if the state recurrence was rolled out indefinitely. When trained for long enough, the model overfits to this subset and performs poorly on long sequences because it encounters unexplored state distributions.</em> -->

## Interventions to Enable Length Generalization
<div style="text-align: justify; margin-bottom: 1em;">
The unexplored states hypothesis indicates that length generalization can be achieved not by changing the architecture or its mechanisms, but by training the model on a more diverse set of state distributions&mdash;in particular, on the distributions that arise when rolling out the state recurrence on long sequences. To do so, we could directly train the model on longer sequences, but this might not always be possible due to GPU memory constraints or due to lack of sufficiently long training sequences. 
</div>

> #### The recipe to achieve length generalization: interventions on the initial state
>
> <div style="text-align: justify; margin-bottom: 1em;"> Most modern architectures assume a zero initial state ($h_{-1}=0$). In our work, we consider four simple interventions on the <strong>initial state</strong> $h_{-1}$, which increase the diversity of states that the model explores during training without the need of training on longer sequences.</div>
{: .block-tip}

<!-- Note that using a non-zero initial state can be interpreted as starting to process the sequence with some previous context. -->

<div style="text-align: justify; margin-bottom: 0.5em;">
The four training interventions can be seen as sampling the initial state $h_{-1}$ from four different distributions that progressively get closer to the distribution of attainable states:
</div>
<div style="text-align: justify; margin-bottom: 0.5em;">
1. <strong>Random Noise</strong>: The state is initialized with an IID Gaussian with zero mean and a constant standard deviation (using the same mean / standard deviation for all layers and heads).
</div>
<div style="text-align: justify; margin-bottom: 0.5em;">
2. <strong>Fitted Noise</strong>: During training, we record the mean and standard deviation of the final states of the sequences across all layers and heads. Then, we initialize the state with an IID Gaussian distribution with mean and standard deviation fitted to the ones seen during training (using a different mean / standard deviation for each layer and head).
</div>
<div style="text-align: justify; margin-bottom: 0.5em;">
3. <strong>State Passing (SP)</strong><sup id="fnref1"><a href="#fn1">1</a></sup>: We use the final state of a previous (unrelated) sequence as the initial state. These final states are obtained by applying the state recurrence on a given sequence, <em>attaining</em> $h_T$ and using it as $h_{-1}$ for another sequence. This is similar to what happens at validation: the model doesn't stop at $T$, but rather keeps rolling the state and producing outputs from $h_T$. 
</div>
<div style="text-align: justify; margin-bottom: 0.5em;">
4. <strong>Truncated Backpropagation Through Time (TBTT)</strong> <d-cite key="TBTT_1990"></d-cite> <d-cite key="TBTT_sutskever"></d-cite>: In this case, we split a long sequence into smaller chunks, and use the final state of each chunk as the initial state of the next one. This is equivalent to processing the whole sequence, yet stopping the gradient propagation between chunks.
</div>

{% details Difference between SP and TBTT  %}
For simplicity, we implement SP by using the final state of the previous batch of sequences as the initial state of the new one. Thus, in practice the only difference between SP and TBTT is that TBTT requires carefully setting up the dataloader so that the sequences of the previous batch correspond to the prior parts of the sequences in the new batch.
{% enddetails %}




<div style="text-align: justify; margin-bottom: 1em;">
The following figures show the results of post-training the official Mamba-2 models for 500 steps (~0.1% of pre-training budget) with each intervention:
</div>

{% include figure.liquid loading="eager" path="assets/img/2025-07-06-length-generalization/interventions_2.png" %}

{% include figure.liquid loading="eager" path="assets/img/2025-07-06-length-generalization/interventions_1.png" %}

### Takeaway #1: SP and TBTT enable length generalization

State Passing and TBTT -- which are the interventions that are closer to realistic states -- allow length generalization in sequences much longer than those seen during training.
Thus:

> #### Takeaway
> <div style="text-align: justify; margin-bottom: 0.5em;"> Length generalization is expected to be <strong>readily achievable in recurrent models</strong> through <strong>simple training interventions</strong>.</div>
{: .block-tip}

Note that our results were achieved *with only ~0.02% of the original pre-training budget*!

### Takeaway #2: Properties of the state of recurrent models

> #### Takeaway
> <div style="text-align: justify; margin-bottom: 0.5em;"> We can infer properties of the <strong>distribution of the state</strong> of recurrent models by looking at the <strong>performance of the interventions</strong></div>.
{: .block-tip}

<div style="text-align: justify; margin-bottom: 1em;"> 
The Random Noise intervention fails to length generalize in the 370m, whereas Fitted Noise works. This suggests that for the 370m model the distribution of attainable states cannot be approximated with a Gaussian with fixed variance, but it can be approximated with an IID Gaussian with fitted variance in each layer and head of the state. However, the Fitted Noise intervention fails to achieve length generalization in the 1.3b model, indicating that the state of large models probably has complex dependency relationships among its elements and thus cannot be approximated with IID values.
</div>



<div style="text-align: justify; margin-bottom: 1em;">
Additionally, the interventions also fix the increasing state norm behavior we showed before, by making the model output states with similar norm at all timesteps: 
</div>
<div style="max-width: 400px; margin: 0 auto; text-align: center;">
{% include figure.liquid loading="eager" path="assets/img/2025-07-06-length-generalization/statemetrics_full.png" %}
</div>
{% details SP in prior works  %}
<p id="fn1">
  <sup>1</sup> Prior works have used the State Passing technique <d-cite key="longssm-wang2024longssmlengthextensionstatespace"></d-cite><d-cite key="end_to_end_bansal2022end"></d-cite>, yet it was applied to different recurrent architectures (e.g. time-invariant ones) or to tasks different to text modeling. To the best of our knowledge, we are the first to show that this technique used as a training intervention can greatly improve the length generalization of several recurrent models, and that it is as effective as TBTT in text modeling.
  <a href="#fnref1">↩</a>
</p>
{% enddetails %}

## Performance on Long Context Tasks
<div style="text-align: justify; margin-bottom: 1em;">
We have seen that the interventions enable length <em>robustness</em> (i.e. not having decreased peformance after the training context $T$), but it is not clear whether they enable length <em>generalization</em> (i.e. solving tasks that require exploiting relationships between tokens that are separated by more than $T$ positions). One may wonder whether the interventions enable length robustness by simply preventing the model from reasoning beyond the training context length&mdash;similar to sliding window attention, which can't reason over tokens separated by more than the sliding window&mdash;in which case the models would have constant performance for all evaluation contexts $t > T$, but could not solve tasks that require long context reasoning. In our work we show that <strong>the interventions do enable length generalization</strong> by showing results on three long context tasks.
</div>

<div style="text-align: justify; margin-bottom: 1em;">
<strong>BABILong</strong><d-cite key="babilong"></d-cite>. BABILong is a challenging benchmark which tests both the common sense understanding of a model as well as its ability to capture long range dependencies in text. In the figure below it can be observed that <strong>State Passing enhances the length generalization of the model in both the few-shot and finetuned settings</strong> (we recall that the model is trained and finetuned on sequences of length 2048). Therefore, State Passing is not only useful in fixing the diverging perplexity of established language models, but also in enhancing their ability to solve long context reasoning tasks.
</div>

{% include figure.liquid loading="eager" path="assets/img/2025-07-06-length-generalization/babilong.png" %}

<div style="text-align: justify; margin-bottom: 1em;">
<strong>Passkey retrieval</strong><d-cite key="landmark_attention_mohtashami2023randomaccess"></d-cite>. The passkey retrieval task requires the model to retrieve a 5-digit passkey inserted at a given depth of a long context. In the figure below we show the performance of the Mamba-2 370m and 780m official checkpoints in three settings: zero shot, regular finetuning, and finetuning with fitted noise<sup id="fnref2"><a href="#fn2">2</a></sup>. The models finetuned with fitted noise are capable of exploiting relationships between tokens that are much more than 2048 positions apart (the training context length). In particular, <strong>the 780m model can solve the passkey perfectly for sequences of length 256k</strong>.
</div>


{% include figure.liquid loading="eager" path="assets/img/2025-07-06-length-generalization/passkey_2.png" %}

{% details Choice of intervention for passkey retrieval %}
<p id="fn2"> <sup>2</sup> Contrary to typical language modeling datasets, the distribution of tokens in the passkey task is not stationary (in other words, there is not a well defined behavior for what the model should do after revealing the passkey). This is why we show results for the fitted noise intervention, as it does not require using the final state of a sequence (i.e., right after revealing the passkey), which might not be appropriate as the initial state.. <a href="#fnref2">↩</a> </p> 
{% enddetails %}

<div style="text-align: justify; margin-bottom: 1em;">
<strong>Synthetic Copying</strong><d-cite key="transformers-better-copying-pmlr-v235-jelassi24a"></d-cite>. The synthetic copying task  consists in copying an arbitrary sequence of tokens. In the table below we show that using State Passing during training greatly improves the validation performance in sequences more than three times longer. Thus, <strong>state passing helps the model length generalize, solving long context tasks that are harder than those seen during training</strong>. 
</div>

{% include figure.liquid loading="eager" path="assets/img/2025-07-06-length-generalization/synthetic_copying.png" %}

## A Deeper Look into How Recurrent Models Process Context
<div style="text-align: justify; margin-bottom: 1em;">
We have shown that the interventions on the initial state enable length robustness and allow solving long context tasks. On top of these findings, we now present a metric that sheds light on how sequence models process their context.
</div>

<div style="text-align: justify; margin-bottom: 1em;">
Ideally, in the case of text modeling we would like the model to pay attention to the recent context, and not focus too much on tokens that are too far away. But how can we quantify this behavior? We introduce <strong>Effective Remembrance</strong> to measure <strong>how much an autoregressive model is "effectively" remembering previous tokens</strong>. Denote by $q(\cdot \| \text{context})$ the probabilities that an autoregressive sequential model outputs for the next token given a context. Then, we define:
</div>
<div style="text-align: center;">
 $ \text{EffRem}_T(t) = d(q(\cdot | x[0:T],q(\cdot | x[t:T])) $
</div>
<div style="text-align: justify; margin-bottom: 1em;">
Where \( d(p,\bar{p}) \) is a distance between probability distributions (e.g., Total Variation). 
\(\text{EffRem}_T(t)\) roughly measures how much the model "effectively remembers" the tokens 
\( x[0:t-1] \) at time \( T \). If \( \text{EffRem}_T(t) = 0 \), this means that the predictions using 
\( x[t:T] \) and using \( x[0:T] \) are the same, meaning that 
<strong>the model does not "effectively remember" any of the past tokens \( x[0:t-1] \)</strong>. 
Conversely, if \( \text{EffRem}_T(t) \) is high, 
<strong>the model is substantially influenced by the tokens \( x[0:t-1] \)</strong>, 
since removing them from the context changes the prediction significantly.
</div>

<div style="text-align: justify; margin-bottom: 1em;">
The following figure shows $\text{EffRem}_T(t)$ for two official Mamba-2 checkpoints (<strong>which fail to length generalize</strong>) for varying $t$ and $T=8192$ (four times the training context):
</div>

<div style="max-width: 500px; margin: 0 auto; text-align: center;">
{% include figure.liquid loading="eager" path="assets/img/2025-07-06-length-generalization/effrem_undesirable.png" %}
</div>

<div style="text-align: justify; margin-bottom: 1em;">
Intuitively we would expect that while every token contributes to the model’s output, the most recent tokens should have a significantly stronger influence. However, notice how the $\text{EffRem}$ curves immediately jump up and then gradually taper off. This behavior is clearly problematic: the next-token prediction at time $T=8192$ shouldn't change drastically depending on whether the model sees only the recent tokens  \( x[4096:8192] \) or the full sequence \( x[0:8192] \). In natural language, the model should primarily rely on recent context, and earlier tokens \( x[0:4096] \) shouldn't completely alter the prediction&mdash;especially not to the extent that the total variation between the two output probability distributions approaches 1. This means that the model is disproportionately influenced by tokens at the beginning of the sequence.
</div>

> #### Intuition
> <div style="text-align: justify; margin-bottom: 1em;">We hypothesize that when a model is always trained with a zero initial state, it uses the <strong>first few tokens it sees</strong> to rapidly differentiate the state, which in turn causes <strong>overfitting to these tokens</strong>.</div>
{: .block-tip}

### State Passing fixes Effective Remembrance
After post-training with State Passing, the $\text{EffRem}$ curves show a gradual increase, indicating that the model places minimal weight on distant tokens and places progressively more weight on recent ones. In particular, tokens in the immediate context (e.g. the previous words in a sentence) have a critical impact on the next token predictions, which is the desired behavior in text modeling.

<div style="max-width: 500px; margin: 0 auto; text-align: center;">
{% include figure.liquid loading="eager" path="assets/img/2025-07-06-length-generalization/mamba2-effrem-reduced.png" %}
</div>

> #### Takeaway
> <div style="text-align: justify; margin-bottom: 1em;"> Through Effective Remembrance, we can check that <strong>State Passing helps the models prioritize recent context</strong> and not be needlessly disrupted by tokens that are far away in the past.
{: .block-tip}




<!-- Lastly, we note that when models fail to length generalize, it is not that the state cannot remember all information from the sequence; rather, in a sense it is so expressive that early elements can completely change its prediction (which is not desirable, as the prediction should mostly focus on the recent context).  Thus, the intuition that the model fails to length generalize because it is not expressive enough to take into account is not correct. The failure to length generalization is related to the models overfitting to early part of the sequences, rather than not being expressive enough. -->

## Conclusion
<div style="text-align: justify; margin-bottom: 1em;">
We have shown that <strong>length generalization is expected to be achievable in recurrent models</strong> through simple training interventions, without the need of changing the architecture nor the internal mechanisms of the model. Moreover, these interventions <strong>improve their performance on long context reasoning tasks</strong>, suggesting that existing recurrent models are not realising their full potential and can be easily improved. 
</div>

<div style="text-align: justify; margin-bottom: 1em;">
Secondly, we believe that this work has significant implications for architecture research. For example, it has become very popular for modern recurrent architecture works to compare out-of-length extrapolation abilities <d-cite key="rwkv-v6-peng2024eaglefinchrwkvmatrixvalued"></d-cite><d-cite key="gated_delta_net_yang2024gateddeltanetworksimproving"></d-cite><d-cite key="beck2024xlstm"></d-cite>. In our work we show that <strong>simple training interventions substantially improve length generalization across several recurrent architectures</strong>, and thus research can focus mostly on the in-length performance (or if directly studying length generalization, it would be important to account for these interventions).
</div>


<div style="text-align: justify; margin-bottom: 1em;">
Lastly, <strong>we have proposed Effective Remembrance as a tool to understand how any autoregressive sequence model processes its context</strong>, thus making it easy to quantify how much models are "effectively remembering" parts of the context.
</div>










