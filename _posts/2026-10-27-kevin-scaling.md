---
layout: distill
title: Some Thoughts on Architecture Research
description: Scaling, academia, and a few open questions.
tags: []
giscus_comments: false
date: 2026-10-27
featured: false

authors:
  - name: Kevin Y. Li
    url: https://www.kevinyli.com
    affiliations:
      name: CMU

bibliography: kevin-scaling.bib

# Keep these names in sync with the section headings below.
toc:
  - name: Some context
  - name: What is compute efficiency?
    subsections:
        - name: So what, why care?
  - name: Compute efficiency in practice
    subsections:
        - name: Fitting the baseline
        - name: Compute efficiency across changes
  - name: Architecture research in academia
---

I intend this piece to be a more informal, stream-of-conscious style of writing.

For some reason -- maybe partially to procrastinate working on my thesis proposal<d-footnote>If you are reading this, I have passed.</d-footnote>, or partially to spew my opinions into the ether in some subconscious effort to immortalize myself in model weights -- I finally decided to reflect, gather my thoughts, and write a blog.
Hopefully, the pretraining data-filter deems my writing *high quality* enough not to be pruned before making it into the gradients.

## Some context

I spent this most recent summer working on pretraining, splitting time between sunny MTV and less-than-sunny SF, although in all transparency, not on architecture-related research.
I have returned to Pittsburgh a lot more opinionated than when I left, much to the annoyance of some of my friends.
Many of my takes revolve around the role of academia and research.
Since most of the past few years of my life has revolved around architecture research, I, naturally, also have some opinons on it.
While I won't put any of the more spicer ones in writing, i.e., in this blog, as not to ruffle any more feathers<d-footnote>You can, however, find me in Sydney for NeurIPS, in which I will be more than happy to divulge.</d-footnote>, I will discuss one important metric/concept that I think should be used more in academic architecture research: compute efficiency (also referred to as efficiency gain, or compute multiplier, you get the idea).

If you've discussed pretraining research with me anytime in the past two, three months, I've probably mentioned compute efficiency.
I probably also recommended you read the [MAI tech report](https://microsoft.ai/pdf/mai-thinking-1.pdf), which I think anyone who wants to get a sense of how industry makes pretraining changes should read.

**Note from future Kevin**: Reflection's [Beam model technical report](https://reflection.ai/documents/beam-model-tech-report) is also a great read, especially their "Scaling Methodology" section, which overlaps with some of the content in this blog.

## What is compute efficiency?

**Compute efficiency** (CE) itself is a metric of how compute effective an architectural change is.
Concretely, it measures the relative compute a baseline would need to reach the same performace of a proposed change <d-cite key="gundlach2025originalgorithmicprogressai,microsoft2026mai"></d-cite>.

To estimate the amount of compute a baseline would need, one could train a very granular assortment of baseline configurations of varying model scale, *but* researchers have found that the performance of models<d-footnote>Because we are discussing pretraining, the usual <em>one true metric</em> is pretraining loss. However, there are some caveats to this that I will discuss later.</d-footnote> usually follows a predictable trend, hence scaling laws.
Thus, a standard power-law is fit to the losses of a baseline across its scaling ladder, i.e., at multiple model scales, each with some fixed token-per-parameter (TPP) ratio.

To estimate the baseline compute needed, a standard power-law scaling curve is fit to its performance $L(C)$, i.e., pretraining validation loss, across compute budgets $C$ (usually FLOPs):
The relationship between the performance $L(C)$ and compute $C$ can be modeled as

$$
\begin{equation}
\label{constrained-learning:eq:power_law}
\begin{aligned}
    L(C) = E + \frac{A}{C^\alpha},
\end{aligned}
\end{equation}
$$

where $E$ is the irreducible loss, $A$ is the scaling coefficient, and $\alpha$ is the scaling exponent.

With the baseline set, we can now evaluate other setups to it.
The general idea is that a better setup should be able to reach the baseline's performance with less compute; that extra compute can then be saved or used to further train the better variant.
Thus, to estimate the effiency of a run that achieves $L^{\prime}$ loss with $C^{\prime}$ compute, we can simply invert \eqref{constrained-learning:eq:power_law} to determine the baseline compute needed to reach $L^{\prime}$.
The compute efficiency $G(L^{\prime})$ of the run is the ratio of the 
The run's compute efficiency is the ratio of the baseline compute to actual compute used:

$$
\begin{equation}
\label{eq:ce}
G(L') = \frac{L^{-1}(L')}{C'} = \left(\frac{A}{L' - E}\right)^{1/\alpha}\,\frac{1}{C'}.
\end{equation}
$$

Here, a "positive" CE of $1.2$ indicates the baseline would require $20\%$ more compute than used to reach the same performance.
Conversely, a "negative" CE of $0.8$ indicates the baseline would only need $80\%$ of the compute used.
In general, CE can be slightly noisy, and I've heard that a CE between $0.95-1.05$ is generally seen as a wash.
This sentiment is also supported by the most recent Beam report out of Reflection <d-cite key="reflection2026beam"></d-cite>, and can be seen in our plot below where the compute efficiency of the baseline still deviates from the ideal 1.0 value.

{% details Why fix TPP? %}

Chinchilla scaling laws found that the optimal allocation of compute increased model parameter count and token budget proportionally <d-cite key="hoffmann2022chinchilla,pearce2024reconciling"></d-cite>.
Thus, holding TPP fixed keeps the relative allocation of parameter count to token count comparable across scales.
And while the original paper suggested that 20 TPP was the optimal ratio, we live in a work where models are overtrained for many reasons, e.g., serving constraints, infrastructure limitations for larger models, etc <d-cite key="sardana2024beyond,grattafiori2024llama"></d-cite>.
So, what should the TPP be fixed to?

The [second MAI pretraining report](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/), which I also highly recommend, seems to fix the TPP to 625<d-footnote>Based on final 50T tokens at 80B active parameters.</d-footnote>, as their scaling ladders were used to estimate the final “hero run”, aka final model.
I would imagine other labs are doing something similar.
Fortunately, the scaling of overtrained models is also predictable <d-cite key="gadre2024reliable"></d-cite>.

I, personally, think that it is completely reasonable for academia to work with 20 TPP.

{% enddetails %}

### So what, why care?

Now there might be the question: why does this matter, can't we just directly compare the losses of the baselines and comparisons?
The power of CE comes less from an evaluation at a single point but more from the trends that arise when comparing across a scaling ladder.
For instance, a flat, positive CE across multiple model scales would suggest that the change would probably hold at larger scales and would be good to incorporate.
However, a positive CE that declines with compute would be a cause for concern, as it suggests that the benefits would vanish after a certain point.

## Compute efficiency in practice

Let's look at a real-life example of CE (not to be confused with cross-entropy!) in action.
Here is the scaling plot from Ricursive's Ripple-65B release <d-cite key="ricursive2026ripple"></d-cite>.

{% include figure.liquid
  path="assets/img/2026-10-27-kevin-scaling/ricursive_scaling.png"
  alt="Scaling curves of Ripple ladders on dolma_stack."
%}

In the figure, we see a variant of compute efficiency at work.
Instead of determining the amount of compute required by the baseline to reach the *Final* Ripple point, here it is inverted where it is estimating the amount of compute it would take the *Final* Ripple recipe to reach the baseline performance.
These style of plots are quite common, but at least for me, are generally a bit hard to decipher especially when I'm trying to determine the marginal utility of changes.
I attribute my struggles comparing the efficiency changes across rungs of the scaling ladder, maybe due to the x-axis log-scale distorting gaps between curves; but maybe this is a skill issue on my end ¯\\_(ツ)_/¯.

### Fitting the baseline
Using Ripple v1.0 as the baseline<d-footnote>I didn't use the Olmo3 runs as the baseline on purpose, as they gave a poor fit. This might be due to the non-optimal training hyperparameters: something the Ricursive blog alludes to based on the batch size and learning rate adjustments made between Olmo and v1.0.</d-footnote>, we can fit a power law to its reported losses<d-footnote>We fit the curve using Huber loss with L-BFGS-B.</d-footnote>.
{% include ripple-chart.liquid part="fit" %}

> **General note on fitting.** The power law is usually **not** fit on all points within the scaling ladder but on some subset. The remaining points are used to check the extrapolation of the fit.

In our case, for the sake of demonstration and getting the best fit (potentially at the cost of extrapolation), I used all six v1.0 points to fit the curve.
If I had fit only on the first five points and predicted the performance of the final rung of the ladder, I would have underestimated the loss by 0.0217, or by a 1.8%.
This is not ideal.
Regardless, let's move onto the second part: calculating compute efficiency!

### Compute efficiency across changes
Following Equation \eqref{eq:ce}, we can plot out the compute efficiencies at each of the subsequent ladders and the original Olmo baseline and analyze the trends.
While the default plot displays the compute efficiency of each version relative to Ripple v1.0, we also have a variant which baselines each ladder to the prior version.
The caveat here is that our fits are based on performance on Code `dolma_stack`, which is not reflective of the training distribution and probably not the only metric used.

{% include ripple-chart.liquid part="efficiency" %}

Our plot also shows just how important training data is, something that should have also be pretty clear in the original Ricursive plot as well.
The changes between ladders v1.1 to v1.6 improve compute efficiency to around 1.8 of v1.0, but the new data mixture immediately increases it to more than 7 (more than 4 relative to v1.6), with increasing gains with more compute.
It really also shows how detrimental non-optimal hyperparameters, such as batch size, learning rate, etc., can be when it comes to training larger and larger models: the compute efficiency of OLMo relative to v1.0 takes a nose-dive at around $10^{20}$ FLOPs.

The compute efficiency plot also directly supports some of the decisions made by the team when moving from version to version.
Between v1.1 and v1.2, we see that additional pre-norm is neutral at small compute budgets but gives increasing gains with more compute.
The introduction of global NoPE and learning rate cooldown (v1.5) also provides increasing gains.
Thus far, our compute efficiency plots line up well with architecture and optimization decisions made across versions; however, the remaining changes seem to have a bit less correlation with our trends.

*However*, I would trust their decisions over our compute efficiency plots, as I believe that the compute efficiency trends would line up with the final decisions if the dataset used was more comprehensive than what was reported in the blog.
But let's still take a look at the remaining changes.
The compute efficiency trend of the sigmoid gated attention is a bit more noisy, and v1.4 (mHC) seems to show pretty consistent gains on top of v1.3.
Finally, the gains of the value residuals seem to be diminishing as the gain decreases from 1.25 to an estimated 1.09 across the ladder.
Again, I would like to reemphasize that these values are based on one dataset and don't show the full picture.


## Current architecture research in academia

Now that we've covered how we should evaluate modeling changes, I'm going to briefly discuss general trends I see in architecture research, especially on the academic side, and my thoughts on them.

If you look at most architecture research papers on arXiv, the evaluation setup is pretty standard: train 125M, 350M, 760M, 1.3B models on some fixed token count, commonly 100B, of your favorite dataset, such as Fineweb-Edu.
I'm not exactly sure where this trend exactly started, but it seems to have taken hold at least a few years ago.
For instance, models introduced in 2023, such as Pythia <cite https://arxiv.org/abs/2304.01373>, RWKV <https://arxiv.org/abs/2305.13048>, and Mamba-1 <https://arxiv.org/abs/2312.00752> all trained on 300B tokens of the Pile, and RetNet <cite https://arxiv.org/abs/2307.08621> trained on 100B tokens.
Even papers not directly focused on architecture, but more generally pretraining, used 100B as the token budget <cite https://arxiv.org/abs/2205.10487>.
My hypothesis is that this is more of artifact of pre-scaling pilled language model research where the intuition was to use *all* the training data for each model scale.

My personal opinion is that this approach to validating performance is quite outdated in the age of scaling.
The reason is directly tied to why scaling laws and scaling research has been so prevalent: we want to predict the performance of a certain model before dedicating massive amounts of effort and resources into pretraining it.
Assume there was some magical device which took in as input experimental configurations, e.g., architecture design, optimizer setup, training horizon, training data, etc., and was able to immediately, with no cost, produce the model parameters.
In this world, I don't think anyone would care about scaling laws, everyone would just sweep across all possible conceivable hyperparameter combinations on all the data they had to try to find the configuration that resulted in the best model.
However, such a device doesn't exist, which is why scaling laws are fit to reduce the cost of reaching a great final model.

Under this desiderata of predictability, current convention of evaluating architecture changes leads to a critical issue --- keeping performance predictable.
By training models across scale with the same token budget, each model will vary; larger models might be undertrained, while the smaller ones will be extremely overtrained <d-footnote> </d-footnote>

caveat that if you were to fit a scaling law on all models trained with the entire 


So now that we've covered how we should be evaluating modeling changes, I'm going to gripe a bit about current architecture research, especially done in academia.

not fixed tpp. 
based on past work of dl where train on entire dataset.

make sure to use tpp for non-embedding parameters. made this mistake before.

more reading: https://openathena.ai/blog/delphi/
