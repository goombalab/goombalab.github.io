---
layout: distill
title: State Space Duality (Mamba-2) Part IV - The Systems
description: 
tags:
giscus_comments: false
date: 2024-05-31
featured: false
thumbnail: assets/img/2024-05-31-mamba-2/mamba_tp.png

authors:
  - name: Tri Dao
    url:
    affiliations:
      name: Princeton
  - name: Albert Gu
    url:
    affiliations:
      name: CMU

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
  - name: Systems and Scaling Optimizations
    subsections:
      - name: Tensor Parallelism
      - name: Sequence Parallelism
      - name: Variable Length
  - name: Results
  - name: Future Directions

---

1. [Part I - The Model]({% post_url 2024-05-31-mamba2-part1-model %})
2. [Part II - The Theory]({% post_url 2024-05-31-mamba2-part2-theory %})
3. [Part III - The Algorithm]({% post_url 2024-05-31-mamba2-part3-algorithm %})
4. Part IV - The Systems


Transformers have benefited from 7 years of systems optimization from the whole research community and large companies. The SSD framework draws connections between SSMs and attention, and allows us to implement many of these optimizations for models like Mamba-2 as well. We focus on tensor parallel and sequence parallel for large-scale training, as well as variable-length sequences for efficient finetuning and inference.

## Systems and Scaling Optimizations

### Tensor Parallelism

{% include figure.liquid loading="eager" path="assets/img/2024-05-31-mamba-2/mamba_tp.png" title="Mamba-2 Tensor Parallelism" %}

One difficulty with large-scaling training of Mamba-1 using tensor parallelism (TP) is that it requires 2 all-reduces per layer, compared to just 1 all-reduce per attention or MLP layer in Transformer. This is because some of the SSM parameters are functions of the inner activations, not of the input to the layer. In Mamba-2, with the “parallel projection” structure, all SSM parameters are functions of the input to the layer, and we can easily apply TP to the input projection: 
We split the input projection and output projection matrices into 2, 4, 8 shards, depending on the TP degree.
We use a grouped norm with number of groups divisible by the TP degree, so that normalization is done separately per GPU.
These changes result in 1 all-reduce per layer, instead of 2.


### Sequence Parallelism

{% include figure.liquid loading="eager" path="assets/img/2024-05-31-mamba-2/mamba_cp.png" title="Mamba-2 Sequence Parallelism" %}

When training on very long sequence length, we might need to split along the sequence length and assign different parts to different devices. There are two main forms of sequence parallelism (SP):
For the residual and normalization operation: this replaces the all-reduce in TP with a reduce-scatter, residual + normalization, then all-gather. Since Mamba-2 uses the same residual and normalization structure as Transformer, this form of SP applies directly with no modification.
For the attention or SSM operation, aka context parallelism (CP). For attention, one could use Ring attention to split it up along the sequence dimension. For Mamba-2, the SSD framework comes to our help once again: using the same block decomposition, we can have each GPU computing its local output and its final states, then pass the states between GPUs (using send/receive communication primitives), before updating the final output of each GPU.



### Variable Length
For finetuning and inference, in the same batch we often have sequences of different lengths. For Transformer, one would usually pad so all sequences have the same length (wasting computation), or implement attention specifically for variable length sequences with careful load-balancing. 
With SSM, we can simply treat the whole batch as a long “sequence”, and avoid passing the states between different sequences in the batch by setting the state transition $A_t$ to 0 for tokens at the end of each sequence.

## Results

How well do these optimizations work? The faster SSD algorithm allows us to increase the state dimension ($\mathtt{N}=64$ or $128$ compared to $\mathtt{N}=16$ in Mamba-1).
Even though technically Mamba-2 is more restricted than Mamba-1 for the same $\mathtt{N}$, the larger state dimensions generally improve model quality.
Here we show results for models trained on 300B tokens on the Pile, with Mamba-2 outperforming Mamba-1 and Pythia.

{% include figure.liquid loading="eager" path="assets/img/2024-05-31-mamba-2/blog_lm_downstream.png" title="Downstream Evaluations" caption="Standard downstream evaluations for open source models trained on the Pile" %}

What about **hybrid models**? We have seen from recent and concurrent work (such as [Jamba](https://arxiv.org/abs/2403.19887) and [Zamba](https://arxiv.org/abs/2405.16712))
that combining Mamba layers with attention layers can improve over pure Transformer or Mamba.
We validate at 2.7B parameters and 300B tokens scale that a hybrid model with just 6 attention blocks (and 58 SSD blocks) outperforms 64 SSD blocks, as well as our standard Transformer++ baseline (32 gated MLP and 32 attention blocks).

{% include figure.liquid loading="eager" path="assets/img/2024-05-31-mamba-2/blog_hybrid.png" title="Downstream Evaluations for Hybrid Models" caption="Downstream evaluations for hybrid Mamba/attention models" %}

We also validated that the SSD algorithm is significantly faster than the selective scan algorithm from Mamba-1 for the same state dimension,
and scales much better computationally to larger state dimensions.
Getting those tensor cores to go brrr is the key!

{% include figure.liquid loading="eager" path="assets/img/2024-05-31-mamba-2/ssm_ssd_dstate.png" title="Mamba-2 Efficiency Benchmarks"  caption="Efficiency benchmarks on sequence length 2K" %}

## Future Directions

With SSD, we have connected (linear) attention and SSMs, allowing us to design faster algorithms and implement system optimizations for SSMs. There are still tons of exciting directions that we (and hopefully the community) want to tackle:
- **Understanding**: hybrid models with a few (4-6) attention layers perform very well, even better than pure Mamba(-2) or Transformer++. What are these attention layers doing? Can they be replaced with another mechanism?
- **Training optimizations**: though SSD might be faster than attention, Mamba-2 as a whole might still be slower than Transformers at short (e.g. 2K) sequence length, since the MLP layers in Transformers are very hardware-friendly. Our implementation of SSD does not specifically take advantage of new features on H100 GPUs, and we look forward to future optimization that would make SSM faster to train than Transformers for large-scale pretraining at 2-4K sequence length.
- **Inference optimizations**: there’s a whole suite of optimizations tailored to Transformers, in particular handling the KV cache (quantization, speculative decoding). How would the inference landscape change if model states (e.g. SSM states) no longer scale with context length, and KV cache is no longer the bottleneck?
