main.py is based on the bellow paper by Google Research: https://arxiv.org/pdf/2501.00663v1


Titans: Learning to Memorize at Test Time

Over more than a decade there has been an extensive research effort of how effectively utilize recurrent models and
attentions. While recurrent models aim to compress the data into a fixed-size memory (called hidden state), attention allows
attending to the entire context window, capturing the direct dependencies of all tokens. This more accurate modeling
of dependencies, however, comes with a quadratic cost, limiting the model to a fixed-length context. We present a new
neural long-term memory module that learns to memorize historical context and helps an attention to attend to the
current context while utilizing long past information. We show that this neural memory has the advantage of a fast
parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its
limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its
ability to memorize the data, acts as a long-term, more persistent, memory. Based on these two modules, we introduce
a new family of architectures, called Titans, and present three variants to address how one can effectively incorporate
memory into this architecture. Our experimental results on language modeling, common-sense reasoning, genomics,
and time series tasks show that Titans are more effective than Transformers and recent modern linear recurrent models.
They further can effectively scale to larger than 2M context window size with higher accuracy in needle-in-haystack tasks
compared to baselines.

In this paper, we present a neural long-term memory that, as a meta in-context learner, learns to memorize at test time.
The neural memory module is a recurrent model in nature, and is adaptively memorizing tokens that are more surprising
or are close to surprising tokens. Comparing to modern recurrent models, it has more expressive memory update and
storing mechanism. Using this memory, we present Titans architectures, and its three variants, in which we suggest to
incorporate the memory module as (1) a context, (2) gating, and (3) a layer. Our experimental evaluation on diverse tasks
tasks validate that Titans are more effective than Transformers and recent modern linear recurrent models, specifically for
long context. That is, Titans can scale to larger than 2M context window size with better accuracy than baselines.
Titans are implemented in Pytorch and JAX and we intend to make the code we used to train and evaluate our models
available soon.
