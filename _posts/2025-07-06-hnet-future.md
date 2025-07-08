---
layout: distill
title: H-Net - The Future
description:
tags:
giscus_comments: false
date: 2025-07-06
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
  - name: Efficiency (and a Connection to Speculative Decoding)
    subsections:
  - name: The Revival of Architecture Research
    subsections:
  - name: Scaling Laws are Horribly, Horribly Broken
    subsections:

---

In this post, I'm going to try to convince you why H-Nets are fundamental and important.
There was only so much content that could make it to the paper. I think there are a lot of downstream consequences and interesting technical connections that we didn't cover.
Much of this will be based on deeper (but mostly unvalidated) intuitions I have and speculative implications about H-Nets
For fun, I'll formulate several concrete hypotheses and predictions about the future of this research direction -- these are personal takes that aren't meant to be calibrated or taken super seriously, but does crystallize what I consider important problems for the field..

If you're an LLM researcher who happened to stumble across this post but doesn't give a crap about architectures, perhaps you'll at least find the last section on [scaling laws] interesting.
If there's one thing I hope this post can influence, it's the community awareness and protocols around this topic.

## Direct Applications

[//]: # H-Nets Will (Hopefully) Proliferate

[talk about miscellaneous implications of HNets]

Let me first say that by "H-Net", I don't necessarily mean our *exact* model,
but the concept we introduced and our definition of H-Net: hierarchical networks with dynamic segmentation strategies.

### Alternative languages and modalities

The biggest use case of H-Net is, of course, languages and modalities that don't have obvious syntactic boundaries for meaningful chunking of the data.
(In contrast, the white spaces of English are heavily used in modern LM pipelines, namely through being hacked into the tokenization pipeline. CITE/LINK)
This means they have much weaker tokenizers that should strongly benefit from using H-Net, which ideally would discover better chunking boundaries.

{% include figure.liquid loading="eager" path="assets/img/2025-07-11-hnet/bpb_chinese_code.png" %}

Indeed, the most striking results in our paper are for non-English languages.
We showed that on both **Chinese and code**, H-Net scales much, *much* better than the standard tokenized language model.
I imagine these results should hold for many other distributions, including **math** (as shown in the SpaceByte paper) as well as **most human languages**.
The linguists have done a much better job than us of explaining [how broken tokenization is for certain languages](https://arxiv.org/abs/2103.06874) <d-cite key="clark2022canine"></d-cite>.


{% details Baselines %}
<span style="color:blue">AG: I'm not sure whether to include this note. Is it too detailed?</span>

I'll note that it's entirely possible (and likely) that the tokenizer we used is suboptimal for those languages, and retraining the tokenizer directly on the data distribution should lead to stronger baseline results.
I think that in the setting, the gap to H-Net would shrink but still be sizable for a few reasons.

1. Languages like Chinese don't have the whitespace heuristic that's baked into tokenization, so their tokenizers are probably weaker than the ones for English.
On the other hand, H-Net is completely free of such priors, so I'd expect the gap on such languages to still be larger than the gap on English.

2. [SpaceByte], one of our most relevant prior works (and criminally unknown IMO) did in fact retrain the baseline's tokenizer on math and code data.
It does indeed improve the tokenized baseline's performance, but still has a noticeable gap to SpaceByte, which in turn is substantially weaker than H-Net as we show.

Of course, the very fact that even trying to make the baseline reasonable requires separately re-processing this part of the pipeline is a strong argument in favor of H-Net!
{% enddetails %}


Other types of sequential data such as **scientific modalities** are hungry for new architectures because the standard Transformer isn't enough (as touched on in my [previous blog post]).
Out of all the experiments we did in this paper, the DNA results seemed perhaps the most striking in terms of the improvement over standard architectures.

{% include figure.liquid loading="eager" path="assets/img/2025-07-11-hnet/bpb_dna.png" %}

I think this most directly validates the power of hierarchical modeling, as the baselines here are also operating on single base pair (non-tokenized) resolution.
On data that isn't pre-compressed by tokenizers,
applying an end-to-end model that tries to learn hierarchical patterns from data seems like a free win!<d-footnote>Once again, I do suspect that DNA could benefit from a BPE tokenizer just like other languages.
I'm not quite sure why I haven't seen anyone use this; it will certainly still lead to a valid DNA language model.
I guess it just seems too weird to the biologists? Well, another point against tokenizers.</d-footnote>

I'll also mention that we didn't spend too much time on DNA and never tried iterating the H-Net to 2-stages.
It might just get better for free, just like for language?

> #### The Future
> H-Nets will immediately be adopted for sequence modalities with unusual characteristics, like tail languages and genomics.
{: .block-tip }

A final type of application that H-Net could unlock is allowing "tokenization" of **continuous-valued sequences**.
The concept of tokenization as applied to language models is predicated on discrete vocabularies -- the process involves explicitly constructing new vocab words.
So it's not applicable directly to non-discrete data like audio and video, even though those modalities seem like they should benefit from a form of dynamic compression (in time) as well!

> #### The Future
> H-Nets will be useful for audio and video, but may require more research.
{: .block-tip }

### Multimodal alignment

Speaking of different modalities, so far I've only talked about training models on a single modality at a time.
The power of dynamic chunking will be even more important when moving to multimodal models that fuse different streams of data together.

In particular, multimodal streams with temporal mismatch (e.g. text and audio) are difficult to synchronize.
Learned chunking mechanisms provide a path to fuse multimodal streams "ticking" at different rates, unlocking native *multi-channel* multimodal models operating at a higher abstraction level.

> #### The Future
> H-Nets will unlock multi-channel multimodal models with temporal fusion.
{: .block-tip }


### Language (and reasoning)

Of course, the big question is how good this model *really* is for core language modeling, which is still at the heart of the most mainstream AI progress today.

I'm personally very bullish on H-Nets -- I wouldn't have worked in this direction if not -- but as always with architectural research, there are significant unknowns, risks, and barriers to adoption.

The main reasons I think they'll improve language modeling has been laid out in the paper,
but in a nutshell, the goal of H-Nets is to *compress data into semantic units* and **operate over higher levels of abstraction**.
It's currently not known to what extent they can successfully do this right now (but I think the scaling behavior we showed is evidence for it),
but if possible, it should just allow for stronger models that are more capable of reasoning intelligently.

> #### The Future
> H-Nets will have increased **language modeling** and **reasoning abilities**.
{: .block-tip }

We didn't formally run out true scaling laws in the paper, which would require sweeping over many more model sizes and compute horizons.
But I think based on the current results, and if my intuition is correct, H-Nets should have fundamentally better scaling behavior.

> #### The Future
> H-Nets will display **stronger scaling laws** than current non-hierarchical sequence model architectures.
{: .block-tip }

Given the importance of language and it being our main motivation for this model, the rest of this post will focus exclusively on language modeling intuitions. [and LLM concepts like inference]

## Efficiency and Engineering (and a Connection to Speculative Decoding)

One of the first things that gets asked about any new architecture is how efficient it is.
Much of my prior architecture research has specifically focused on being [more](https://arxiv.org/abs/2111.00396) [efficient](https://arxiv.org/abs/2312.00752) than the status quo.
In the H-Net paper, we basically didn't touch on efficiency, so what can we say about this?

### The efficiency ↔  quality Pareto frontier

Well, the simple reason why we didn't is because the intuition behind our method (chunking) is more obviously connected to model *quality* rather than *efficiency*, so we focused on those aspects.
But of course, what really matters is the *interaction* between efficiency and quality.
This is generally monotone, leading to an entire Pareto frontier of performance tradeoffs.
At a very superficial level, I expect that the quality gains of H-Nets would directly translate to efficiency gains as well.

{% include figure.liquid loading="eager" path="assets/img/2025-07-11-hnet/efficiency_quality.png" caption="This figure hopefully contains zero information content, but <br> since I decided to draw it for some reason I'm not gonna waste it." %}

But there can often be more nuance to this with architecture research because of qualitative differences between models.<d-footnote>For instance, the efficiency ↔  quality tradeoff of Mamba vs. Transformers isn't actually so clear-cut, as discussed in [my previous blog post]).</d-footnote>
What people want to know is whether there are **qualitative inductive biases** in the architecture that directly relate to its efficiency.
(This is what made SSMs so appealing, I suppose, as these biases -- the constant state size -- are very intuitive.)

This topic is a lot more subtle for H-Nets, but there are a few intriguing connections to highlight.

[//]: # Although I will say, this can often be more nuanced when working on architectures because of qualitative differences between models (for instance, the efficiency ↔  quality tradeoff of Mamba vs. Transformers isn't actually so clear-cut, as discussed in [my previous blog post]). But since H-Nets don't touch the core sequence mixing layers for the most part, hopefully, the trade-off is a little more straightforward here.<d-footnote>Although to be honest, I'm not entirely sure of this, because the global hierarchical nature does impact the choice of sequence mixing layers and probably does lead to unexpected qualitative differences; for example, as touched on in [this section]. <span style="color:blue">AG: This might be too distracting, I guess I started rambling.</span>


### Speculative decoding

Let's think about **speculative decoding (specdec)**, which is by now a universally used technique for LLM inference.

{% include figure.liquid loading="eager" path="assets/img/2025-07-11-hnet/speculative.png" caption="The speculative decoding process of stepping a small model on every token and a large model on every few tokens strongly resembles the H-Net decoding process. [<a href='https://arxiv.org/abs/2203.16487'>Source</a>]" %}

#### Speculative decoding resembles an H-Net
Without getting too in the weeds,
speculative decoding consists of a *large model* (usually called the "verification model") that we want to sample from, and a *small model* (usually called the "draft model") that'll help us sample from it faster.
The decoding process basically looks like this:
1. On every autoregressive step, the *draft model* will take a step to generate a new token.
2. Every few steps, the *verification model* will verify the small model's sequence of tokens, accepting as many of them as it can.

At a high level, speculative decoding improves generation speed by letting the large model only do a forward pass every few tokens.
But this is incredibly similar to the decoding process of an H-Net!
1. The H-Net *encoder/decoder networks* take a step on every token
2. The H-Net *main network* takes a step every few tokens (on every *chunk*)

#### Speculative decoding + H-Net is redundant
One can take this a step further and ask: what happens if we combine speculative decoding to try to speed up an H-Net?
In the *ideal* case of speculative decoding, what might happen is:
1. The small model (an auxiliary draft model) takes a few steps, say $k$, and proposes a number of tokens
2. The large model (H-Net) does a forward pass on these $k$ tokens. Suppose that $k$ lines up with the next chunk: then this amounts to
    - $1$ parallel pass (over $k$ tokens) of the H-Net's encoder/decoder networks
    - just $1$ standard decoding step of the H-Net's main network

Further suppose that the draft model has a similar network structure as the H-Net's encoder/decoders.
Then this is almost the same as the vanilla decoding process of an H-Net above, just with the sequential steps of the encoder/decoder swapped for sequential steps of an auxiliary draft model!

So this application of speculative decoding was pretty much a no-op in terms of improving efficiency.

Why is this? One way to interpret this is that the speculative decoding process is already *baked into* the H-Net structure,
and once we move to H-Nets, we might not need inference techniques like specdec anymore.

#### Entropy strikes again
And there's one final conceptual connection!
Speculative decoding works because there's some form of redundancy in token sequences that makes some of them easier to predict than others; the speedup happens exactly when there are small sequences of easy-to-predict tokens.
Or put another way, when there are local strings of low-entropy or low-information tokens.
But this is exactly one of the heuristics for how to think about dynamic hierarchical networks [LINK] -- they segment on surprising tokens, or more generally on some notion of information content in the data.

All in all, there are a lot of striking similarities between speculative decoding and H-Nets!
My hypothesis is that: **the H-Net structure implicitly subsumes the speculative decoding process**.

What are the implications of this?
Well, the obvious practical one is that despite seeming more complicated than standard models, H-Nets might not actually be much more complicated than modern LLM *pipelines* used in practice.

> #### The Future
> H-Net inference engineering will have a similar complexity to current LLM inference pipelines.  
> <br>
> Furthermore, inference tricks will become marginally less and less useful (and ideally not be necessary at all), as they become subsumed by end-to-end models that incorporate the underlying concepts in a more natural way.
{: .block-tip }

But to me, there's a more important conceptual implication.
- Much of architecture efficiency optimization consists of asking: **Given this model, how can I make it faster?**
- One can flip this on its head and ask: **Given that this model *could* be sped up, what does that imply about the original model?**
The way to think about this is that *the very fact that standard LLMs can be sped up* through tricks like speculative decoding means that *the original models have redundancies* and could be further optimized.

The H-Net structure is exactly the way to smooth out those redundancies,
**baking (something akin to) the speculative decoding process directly into the model**,
while **leveraging parameters and compute** better and **training everything end-to-end**.
In other words, the structure of the H-Net preserves the same characteristics of the *inference-optimized* standard LM, but with a better *training objective*.<d-footnote>Just to unpack a bit more: intuitively, the reason this should lead to a stronger model is because the main network (analogous to the target verification model in specdec) is trained directly on *chunk-level* modeling, the way they would be used at inference, instead of the specdec pipeline of being trained on a more granular (*token-level*) objective and being used in a different way at inference.</d-footnote>

### Engineering Challenges

Okay, a lot of what I've talked about so far (and the way it's implicitly described in the paper) is about *theoretical* efficiency;
we considered FLOPs, not wall-clock time.
A very important question is: is the theoretical efficiency realizable in practice?

#### Training

[talk about training efficiency, echoing the paper]

[compare to MoE]

#### Inference
influence has largely been discussed in relation to speculative decoding;
I think it's going to take some work, but don't see any fundamental barriers.

Overall, engineering for H-Nets will be a substantial but surmountable problem for their adoption at scale.

> #### The Future
> H-Nets will require non-trivial **research and infrastructure** work for both **training and inference**; on the order of what was needed for tokenizers, mixture-of-experts, and speculative decoding pipelines.  
> <br>
> This effort will be worth the tradeoff to achieve higher quality and less brittle end-to-end models.
{: .block-tip }


## Revival of Architecture Research

As multi-component pipelines are consolidated into end-to-end models, previous parts of the pipeline that required dedicated treatment will transform into added complexity in the model instead.
Architectures will become somewhat more sophisticated and require new considerations and research. 
Here are a couple of such considerations.

### Hierarchical Sequence Models

As I [started off this series of blogs describing], hierarchy is far from new.

There have been a few recent works that investigate hierarchical structures inside novel sequence model layers (i.e. variants of attention or SSMs).
- [Native Sparse Attention (NSA)] CITE is a recent sparse attention model that performs a 2-step process of aggregating information inside local blocks.
- [Log-linear attention] and [prefix scannable models] introduce new hierarchical layers that generalize modern recurrent models using a binary tree of hierarchies, improving their expressivity by increasing the constant size state to logarithmic.

While these models are elegant exercises in algorithm design and engineering, and definitely valuable contributions to the community,
I think there might be fundamental problems long-term with building hierarchy directly into the layer.
The root cause is the difficulty of having a dynamic or flexible hierarchy.

In NSA, for example, there is a block size hyperparameter (set to $64$ by default, I think) that governs the lower level of the hierarchy, motivated by hardware [alignment].
This doesn't feel "right" to me for some reason. I guess it's because I think that while hardware considerations are important, they should be connected to the *model algorithm* rather than the *model definition*.
For example, while [Mamba-2](https://arxiv.org/abs/2405.21060) also has a *block size* hyperparameter (also set to $64$ by default) related to the size of matmul tiles,
this only affects its implementation/efficiency and not the definition of the model;
in contrast, the block size parameter of NSA fundamentally changes what functions (sequence transformations) it can represent.

As another example, the log-linear models are tied to a static binary-tree hierarchy.
But a major theme of the H-Net paper is that static hierarchies are not the right structure.

> #### The Future
> The best way of building hierarchical models will be in the holistic architecture's **network structure** like H-Net, not in individual layers.
{: .block-tip }

A future direction of H-Nets is to see how far the hierarchy can extend.
For example, one can build a binary tree-like architecture with repeated downsampling by a factor of roughly 2 (a controllable parameter in the H-Net), which leads to the same holistic properties as log-linear layers -- linear-scaling computation in sequence length with logarithmic-scaling state size -- but with a dynamic hierarchy.
By carefully balancing the depth/widths as well, one can actually get very fine control over the scaling of both compute and state size, potentially targeting **polynomial-scaling state sizes** which is [hypothesized to be optimal for language modeling](http://arxiv.org/abs/2503.04725v1) <d-cite key="chen2025l"></d-cite>.

### Long Context

One question I have is whether deep hierarchies can actually allow one to completely get rid of global attention.
What happens if one uses a deep recursive H-Net with pure constant-state-size layers like SSMs or sliding window attention (SWA)?
In a normal isotropic model, these suffer on long sequences because they simply can't encode enough information.
But in a deep hierarchical model, information is constantly being compressed and abstracted,
shrinking the sequence length and perhaps substantially improving the effectiveness of these layers.

Maybe some lightweight global attention will still be needed for certain retrieval abilities?
But I think hierarchical structure can only help long context significantly (which is indeed an explicit motivation for many prior works on hierarchical sequence models!).

> #### The Future
> H-Nets will substantially **improve long context abilities**.
{: .block-tip }

We originally wanted to explore some long context benchmarks in the H-Net paper, but there were too many facets to show already so we didn't get around to it.
Hopefully someone will investigate this in the future!

### Hybrid Models

While hybrid models combining linear layers with quadratic attention have become much more popular,
I always wondered if they were the most natural way.

One nice thing about H-Nets is that they can hybridize linear and quadratic layers in a more elegant way, in my opinion.<d-footnote>In my head, another potential meaning of H-Net stands for *hybrid network*.</d-footnote>
Linear layers go on the outside, both for efficiency *and* inductive bias reasons (as covered in [my previous post]),
and powerful quadratic attention layers can go on the inside, operating over higher levels of abstraction where they are most suited.

However, figuring out the exact right combination of layers is pretty non-trivial.
We did endless ablations over the course of this project (and included many of them in the paper, but that was only a small subset),
and it was pretty hard to come to conclusive answers.

For example, these were the conclusions found for a 2-stage H-Net (three different [sequence lengths]):
- **Outer**: Pure Mamba layers perform best, and seem indispensable.
- **Middle**: After the outer layers have shrunk the sequences by a reasonable length (almost $3\times$), this is much closer to tokenized language, and I wouldn't have been surprised if pure Transformer layers were fine here. But we found that Mamba was still crucial, which validates that its effect is not *just* because it's good at high resolution, but because it's doing a form of **active compression that benefits dynamic chunking**.
- **Inner**: The innermost model has the most parameters and is essentially an standard isotropic language model operating on coarsely tokenized data. In the paper, we stuck to pure Transformers because that was our main baseline.
However, this is completely orthogonal to the rest of the H-Net design; we did experiment a bit and did an ablation showing that general findings for LM architectures still transfer, such as that **hybrid models (we tried 3-to-1 Mamba-to-Transformer) still have noticeably better perplexity**! CITE [INCLUDE GRAPH]

At the very least, moving toward such hierarchical models will necessitate expanding the space of primitives used; I'm pretty sure standard attention is not sufficient.

> #### The Future
> Linear sequence models such as **state space models will become core primitives** of language models, if only for acting as the byte-level interface.  
> <br>
> In turn, research papers on such models should start **incorporating byte-level language modeling** as a standard evaluation.
{: .block-tip }

### It's the Wild West

I have to emphasize again that creating the H-Net was a [fiendishly difficult design problem], and we still don't completely understand how a lot of things work.
I wouldn't be surprised at all if someone came out next week with a simplification of our routing mechanism that was better (well, I'd pretty surprised actually -- but I do expect it to happen at some point).
At any rate, there are so many new axes of variation, knobs to turn, and completely new directions to explore.
Things are just getting started!

> #### The Future
> Linear sequence models such as **state space models will become core primitives** of language models, if only for acting as the byte-level interface.  
> <br>
> In turn, research papers on such models should start **incorporating byte-level language modeling** as a standard evaluation.
{: .block-tip }

[[I hope we experience] a renaissance for architecture research!]

## Closing

Let me return once again to the higher-level question: is all of this actually useful? Are hierarchical models the future?

In this post, I haven't said anything that was actually technical or rigorous, only a loose set of connections and intuitions.
But somehow to me these point to some type of deeper truth.
Something about certain models just "feels right" to me, and H-Nets feel right.

Perhaps the most concrete answer I can give, though, can be summarized by just two points:
1. Hierarchical pipelines are *already* used everywhere, often implicitly.
2. Consolidating them into general, trainable methods is at the heart of AI ([The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)).


> #### Observation
> Existing LLM **pipelines** are already **implicitly hierarchical**, such as  
> (1) the core modeling (tokenizer -- language model) pipeline  
> (2) the speculative decoding (draft model -- verification model) pipeline
{: .block-warning }

> #### The Future
> As we get closer to finding "the right architecture", these explicitly engineered pipelines will be subsumed by an end-to-end model. *Maybe the H-Net?*
{: .block-tip }

Fundamentally, such end-to-end models just *have* to be more versatile and more powerful.
They will learn faster from data and scale better with compute.

> #### The Future
> H-Nets, or some improvement to them, will **warp the scaling laws**.
{: .block-tip }

## Scaling Laws are Horribly, Horribly Broken

Now I'm going to [change tack] away from the H-Net specifically, and talk about byte-level models more generally, and about scaling laws.

Even if you don't care about architectures, you should care about scaling laws.


### Scaling Laws Depend on the Tokenizer

Everyone talks about scaling laws. Not that many people talk about tokenizers.

And there's an embarrassingly obvious fact that took me a long time to realize:
**the scaling law depends heavily on the tokenizer**.

This is completely obvious in hindsight.


> The performance penalty depends predictably on the ratio $N^{0.74}/D$, meaning that every time we increase the model size 8x, we only need to increase the data by roughly 5x to avoid a penalty.

> One notable conclusion in Kaplan et al. (2020) is that large models should not be trained to their lowest possible loss to be compute optimal. Whilst we reach the same conclusion, we estimate that large models should be trained for many more training tokens than recommended by the authors. Specifically, given a $10\times$ increase computational budget, they suggests that the size of the model should increase $5.5\times$ while the number of training tokens should only increase $1.8\times$. Instead, we find that model size and the number of training tokens should be scaled in equal proportions.

The Chinchilla paper barely mentions the tokenizer at all, and definitely not in the context that implies it affects their recommendations.

I don't know if it's just me, but I remember hearing plenty of sentiments along the lines of "Kaplan et al. pioneered the scaling laws but didn't calculate them as accurately as Chinchilla"

...

Is it possible that the different formulas that everyone comes up with is not due to improved models or more careful protocols for calculating the scaling laws, but simply orthogonal changes. 


I'm sure all the big labs that actually do serious pre-training are aware of this. I just find it interesting that it never seems to be made super explicit in the papers.
And speaking of which, if it's obvious that scaling laws are a function of (model, data, *and* tokenizer) -- why are there so many results on scaling laws of models and data but none on the tokenizer?

> Compared to Llama 2, we made several key improvements. Llama 3 uses a tokenizer with a vocabulary of 128K tokens that encodes language much more efficiently, which leads to substantially improved model performance.

Llama 3 even explicitly mentioned that their new tokenizer was very important, and ran serious model/data scaling laws in the paper, but none for the tokenizer. Puzzling.

### Tokenized models can't compare their perplexities

before talking about scaling laws, let's first mention a problem.

Everyone uses negative log likelilood / perplexity as the most important summary metric of the performance of their models.
But these are calculated with respect to *tokens*, a completely arbitrary, non-standardized, abstract unit.
this makes losses difficult between.



[find paper from COLM 2024 about comparing models across tokenizers]

[losses: allow losses to be compared across models]


### BPB isn't even calculated correctly for tokenized models

But wait, it gets even worse.

1. All LLM papers should report bits per byte instead of token perplexities.

2. Scaling laws should be calculated in terms of bytes of data consumed to make them independent of the tokenizer.



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
