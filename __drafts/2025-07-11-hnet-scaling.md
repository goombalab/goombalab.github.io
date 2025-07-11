---
layout: distill
title: H-Nets - Scaling Laws
description:
tags:
giscus_comments: false
date: 2025-07-10
featured: false
thumbnail: assets/img/2024-05-31-mamba-2/mamba-2-V3-transparent.png

authors:
  - name: Albert Gu
    url:
    affiliations:
      name: CMU, Cartesia AI

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
  - name: Direct Applications
    subsections:

---


By early 2025, we already had results for early versions of H-Nets showing that they performed well with training curves crossing over the baseline tokenized Transformer,
and with an even larger gap for the iterated hierarchical model.

This was exactly what I hoped for; I had [always been convinced hierarchical models would be better]({% post_url 2025-07-11-hnet-past %})
and so I was happy but not too surprised when Sukjun first showed this result.

<div style="max-width: 500px; margin: 0 auto; text-align: center;">
{% include figure.liquid loading="eager" path="assets/img/2025-07-11-hnet/bpb_curve_xl_hnet_white.png" width="50%" %}
</div>

I felt that although we didn't run any formal scaling laws, the training protocol seemed fairly reasonable
and it was pretty clear that the model seemed to be scaling better with data.<d-footnote>The protocol was essentially: fix a model size (or equivalently, inference budget), and run the model out with modern constant learning rate schedulers. This more or less resembles how many recent "overtrained" LLMs are trained, I believe.</d-footnote>

But as we got closer to release, I started wondering to what degree we could claim the model scales better, or actually predict the model's scaling behavior.
This post is going to be a collection of speculations of mine based on limited evidence (and quite frankly, limited experience/understanding of scaling laws).
In the academic and startup world, we don't care much about scaling laws because they're far too expensive and I don't think they're necessary to do genuine innovation (hopefully this paper is a case in point!)
So I haven't really given any beyond-superficial thought to them except in the last few days.
I'm sure my intuition isn't great, and I'll say some wrong things here (please feel free to send a note if I do!).
But here we go anyways.


Beyond speculating about H-Net, though, there are a couple of things around scaling laws that have confused me as I've dug in more.
I hope to use this post to also ask some questions and spread some awareness about what seem to be not-so-obvious facts,
and suggest new guidelines and protocols for the community as they think about these topics.

## Scaling Laws

In this post, I'm going to argue that **H-Net will substantially shift the scaling laws**.

Fitting scaling laws usually consists of two separate stages
1. **Compute to loss**: for every compute budget, sweep the model vs. data size and find the one providing optimal loss. Then fit a curve to determine the compute vs. loss trend.
2. **Loss to downstreams**: then, fit a curve to predict downstream performance of the model.

The first step is generally considered more important. 
Conventionally, this scaling law is usually stated as any pairwise power law relationship between data budget, model size, compute budget, or loss, that leads to optimal performance.
(Since $\text{compute} = \text{model size} \times \text{dataset size}$, two of these determine the third.)

### The first scaling laws

For example, the two original [Kaplan et al.](https://arxiv.org/abs/2001.08361) and [Hoffman et al. (Chinchilla)](https://arxiv.org/abs/2203.15556) papers expressed relationships between model size and data size.
From Chinchilla:
> One notable conclusion in Kaplan et al. (2020) is that large models should not be trained to their lowest possible loss to be compute optimal. Whilst we reach the same conclusion, we estimate that large models should be trained for many more training tokens than recommended by the authors. Specifically, given a $10\times$ increase computational budget, they suggests that the size of the model should increase $5.5\times$ while the number of training tokens should only increase $1.8\times$. Instead, we find that model size and the number of training tokens should be scaled in equal proportions.

This paper was particularly memorable because of its simple-to-remember recommendation: *scale model and dataset size together*.
It showed a constant that said that your dataset size (in tokens) should be roughly $20\times$ your model size (in parameters).

What's striking now about these original papers is the phrasing of these results as a form of truth.
The Chinchilla takeaways seemed to be thrown around like gospel a few years ago, and I remember internalizing them as well (for example this $20\times$ number, which I used quite a lot without questioning until recently).
But these conclusions and exact constants can't be taken in isolation: they implicitly depend on all the other parts of the setup like model architecture, data, optimizer and training protocol, and other factors,
all of which might change these conclusions.

### Shifting scaling laws

A more modern formulation often directly relates compute to loss: $L = a \cdot C^b$, where $L$ is loss, $C$ is compute (in FLOPs), and $a$ and $b$ are constants that describe the law.
The goal of pretraining research is then to find improvements,
such as in model architecture, optimizer, or other variations,
that can decrease $a$ or $b$.

Katie Everett has a couple [nice](https://x.com/_katieeverett/status/1925665335727808651) [threads](https://x.com/_katieeverett/status/1926722325073801612)
that enumerate a number of papers that study scaling laws, varying different factors.
Notably, it seems like almost everything doesn't change the exponent $b$, only the constant $a$.
Thus, the log loss $\log(L) = \log(a) + b \cdot \log(C)$ has the form of a line where the slope is pretty much constant as things vary,
and changes in methods seek to lower the constant $a$ which can be interpreted as "shifting the scaling law".

Some of these studied changes include:
- mixture-of-experts (mixture-of-experts vs dense, at various expert granularities)
- recurrent architectures (xLSTM, HGRN-2, Mamba, etc.)
- optimizer (SGD vs. Adam vs. Muon)

What about the data? It seems there a number of results that change the data and find that it affects the exponent.
Different datasets are compared to each other via "loss-to-loss" scaling laws.
I can't say I fully understand what this means; it seems intuitive that data should affect the law the most because these laws are directly in terms of loss over the dataset...
I've also heard by word-of-mouth from those who study these laws that there isn't a standard way to measure changes in scaling laws across datasets.

But even if you keep the same underlying data, there's one more factor of variation that does implicitly affect the data... the tokenizer, of course.

## How Does Tokenizer Affect Scaling Laws?

The tokenizer is known to directly impact your model's performance.
Llama 3, for example, changed almost nothing in the architecture compared to Llama 2, but did change the tokenizer.

> Compared to Llama 2, we made several key improvements. Llama 3 uses a tokenizer with a vocabulary of 128K tokens that encodes language much more efficiently, which leads to substantially improved model performance.

What seems a bit puzzling is that Llama 3 knew this mattered, and ran serious model/data scaling laws in the paper, but none for the tokenizer.

The reason might be that *different tokenizers, like different data sets, seem difficult to compare*.
Even though the underlying data is the same, the tokenizer implicitly changes the dataset for token-based language models, since it directly changes the tokens that the model receives.
So the loss functions are difficult to compare as the tokenizer changes, which might be one reason why people haven't been testing how to change the vocabulary together with the model parameters.

### A better way to compare losses

Unlike the case of actually changing the underlying data, however, there's a pretty simple way to fix this in the case of changing tokenizers.
Just rescale the loss to a *tokenizer-invariant* quantity: in particular, the bits-per-byte (BPB).
This just amounts to dividing your loss (the negative log-likelihood) by a few constants: a factor that converts it to base 2, and the number of total bytes.

It seems to me that there's absolutely no downside to doing this.
The standard loss is completely uninterpretable; its grounded in an arbitrary vocabulary produced by the BPE algorithm and can't be compared across tokenizers.
This simple rescaling has an intuitive interpretation, and can then start transferring across different vocabularies.

For example, this paper claims that [intelligence represents intelligence linearly](https://arxiv.org/abs/2404.09937) <d-cite key="huang2024compression"></d-cite>,
which means they showed that taking a bunch of open source models (with different training protocols and tokenizers) and simply rescaling them to BPB (or BPC, bits-per-character),
shows a strong correlation with various benchmarks.

{% include figure.liquid loading="eager" path="assets/img/2025-07-11-hnet/bpc_correlation.png" %}


### Vocabulary scaling laws
I found only one paper that directly studied scaling laws for standard vocabularies, which found that [larger models deserve larger vocabularies](https://arxiv.org/abs/2407.13623v1) <d-cite key="tao2024scaling"></d-cite>.<d-footnote>There are also others that propose larger interventions, such as decoupling the input and output vocabularies, which also found positive effects from scaling vocabulary size <d-cite key="huang2025over"></d-cite>.</d-footnote>
In order to do this, they had to consider the vocabulary issue, and used a variant of the BPB. 

{% include figure.liquid loading="eager" path="assets/img/2025-07-11-hnet/vocabulary.png" %}

The conclusion seems pretty intuitive and in line with the Llama 3 change.
It seems that this should definitely shift the scaling law; unfortunately, I can't tell at a glance whether it affects the exponent at all, and the paper doesn't seem to have tried to analyze that directly.

### Byte-level modeling

A final hint is given by the [Byte Latent Transformer (BLT)](https://arxiv.org/abs/2412.09871) paper, a recent work closely related to H-Net <d-cite key="pagnoni2024byte"></d-cite>.

There are two sets of different results that they show.

{% include figure.liquid loading="eager" path="assets/img/2025-07-11-hnet/blt_1.jpeg" %}

This first one runs relatively standard scaling laws, which fix an architecture.
One can directly see the log-linear scaling log trends, and they're all at the same slope, as we discussed above; architectures don't seem to change the slope.

{% include figure.liquid loading="eager" path="assets/img/2025-07-11-hnet/blt_2.jpeg" %}

This result took me a while to interpret, but I think shows essentially the same result as our main "scaling" plot:
It simply fixes the model size (in terms of FLOPs, not parameters) and increases the training budget.

.......

This one is a bit trickier to reason about because it's not a standard scaling law and I'm not sure if it's even log-linear.

So what do we know?
- Increasing vocabulary together with the model size is helpful and should shift the scaling law
- BLT (a byte-level hierarchical model like H-Net) Claimed that changing the architecture. 

## Actually, Bits-per-Byte Doesn't Work Either...

It seems to me like quite a headache to compare.

### Warping Scaling Laws

It's intuitively true to me that tokenizers should also warp scaling laws.
- First of all, they directly affect the meaning of the data, which seems to be the one factor that most strongly affects scaling laws.
- Second, by taking the extreme case<d-footnote>I find this to be a very useful principle for reasoning in general</d-footnote>, we know that the setting of byte-level modeling (the simplest possible tokenizer) seems to display very different behavior for different architectures which otherwise scale similarly using standard tokenizers (see my previous post on [MambaByte vs LlamaByte]).

### Tokenized models can't compare their perplexities

Why does no one do this?
Well, it seems to me that the community has a general gaping blind spot around tokenizers.
There are some concrete reasons why, I think.

Maybe the most direct "mechanical" Reason is that changing the tokenizer directly changes the 

before talking about scaling laws, let's first mention a problem.

Everyone uses negative log likelilood / perplexity as the most important summary metric of the performance of their models.
But these are calculated with respect to *tokens*, a completely arbitrary, non-standardized, abstract unit.
this makes losses difficult between.


Why exactly does every paper still report their scaling laws with "loss", A quantity that depends completely on the arbitrary vocabulary produced by the arbitrary tokenizer that they use?



[find paper from COLM 2024 about comparing models across tokenizers]

[losses: allow losses to be compared across models]


### BPB isn't even calculated correctly for tokenized models

But wait, it gets even worse.

1. All LLM papers should report bits per byte instead of token perplexities.

2. Scaling laws should be calculated in terms of bytes of data consumed to make them independent of the tokenizer.



[In some ways, it's actually encouraging to see this result. It shows that there is a large gap between NLL estimates using only the canonical segmentation vs. the true byte-level conversion of a tokenized model. And what that means is that our results showing the gap between HNET vs. a tokenized model is much more believable. In fact, one way to interpret this is that there is an intrinsic inefficiency or *redundancy* in a tokenized model, which manifests as the exponentially large number of possible token segmentations of a given string of text. And the whole theme of my last post was that I believe end-to-end models like H-Nets are better because *they smooth out redundancies in [non-end-to-end] pipelines And allow models to focus their resources (parameters/compute/etc.) 

But wait! Are the claims in the H-Net paper accurate?

[we discovered the results of very very late into the project (in fact, literally just a few days before releasing the paper), after everything was already done and written up.
And these results are pretty unknown and the implications are subtle, I think.
So I wouldn't have felt bad putting out the H-Net paper anyways.

but I hate over-claiming, so I wanted to make sure I actually believed our results and could put them out in good faith.

and I think the interpretation of the BPB plot as the *Realizable sampling BPB*
rather than the *Compression rate BPB* is a pretty reasonable metric for our use case, which is, of course, focused on language models as a *generative* model that can perform fast auto-regressive sampling.

But I don't really know what the correct interpretation is or what the right thing to do is.
Hopefully, the community becomes much more cognizant of this topic, perhaps through this post, and figures out a consensus.


> #### The Future... I hope?
> The community will start normalizing LLM losses/perplexities in terms of **bits-per-byte**. All scaling laws will use these more meaningful quantities.  
> <br>
> *Please 🙏*
{: .block-danger }

> #### The Future... I hope?
> The community will start normalizing LLM losses/perplexities in terms of **bits-per-byte** or related quantities, instead of token-centric notions.  
> <br>
> All **scaling laws** will use these more meaningful quantities.  
> <br>
> ***Please!** 🙏*
{: .block-danger }


Ultimately, I think that **all evaluations should reason about byte-level models**,
- whether explicit: end-to-end in one model, like H-Nets
- or implicit: multi-stage pipeline, like tokenized LMs + a [token-to-byte algorithm]


### Do H-Nets Scale Better?

Fundamentally, such end-to-end models just *have* to be more versatile and more powerful.
They will learn faster from data and scale better with compute.

> #### The Future
> H-Nets, or some improvement to them, will **warp the scaling laws**.
{: .block-tip }

### Acknowledgements

Thanks to Tri Dao for feedback and suggestions on this post.
