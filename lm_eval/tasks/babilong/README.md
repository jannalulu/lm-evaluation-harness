# Babilong

### Paper

Title: Babilong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack
Abstract: https://arxiv.org/abs/2406.10149

In recent years, the input context sizes of large language models (LLMs) have increased dramatically. However, existing evaluation methods have not kept pace, failing to comprehensively assess the efficiency of models in handling long contexts. To bridge this gap, we introduce the BABILong benchmark, designed to test language models' ability to reason across facts distributed in extremely long documents. BABILong includes a diverse set of 20 reasoning tasks, including fact chaining, simple induction, deduction, counting, and handling lists/sets. These tasks are challenging on their own, and even more demanding when the required facts are scattered across long natural text. Our evaluations show that popular LLMs effectively utilize only 10-20\% of the context and their performance declines sharply with increased reasoning complexity. Among alternatives to in-context reasoning, Retrieval-Augmented Generation methods achieve a modest 60\% accuracy on single-fact question answering, independent of context length. Among context extension methods, the highest performance is demonstrated by recurrent memory transformers after fine-tuning, enabling the processing of lengths up to 50 million tokens. The BABILong benchmark is extendable to any length to support the evaluation of new upcoming models with increased capabilities, and we provide splits up to 10 million token lengths.

Homepage: https://github.com/booydar/babilong

### Citation

```
@article{kuratov2024babilong,
    title={Babilong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack},
    author={Kuratov, Yuri and Bulatov, Aydar and Anokhin, Petr and Rodkin, Ivan and Sorokin, Dmitry and Burtsev, Mikhail},
    journal={arXiv preprint arXiv:2406.10149},
    year={2024}
}
```

### Groups and Tasks

#### Groups

* `babilong`: All Babilong tasks


#### Tasks

The benchmark includes 20 reasoning tasks at various context lengths:

**QA Tasks (qa1-qa20):**
* `babilong_qa1`: Single supporting fact QA
* `babilong_qa2`: Two supporting facts QA
* `babilong_qa3`: Three supporting facts QA
* `babilong_qa4`: Two argument relations
* `babilong_qa5`: Three argument relations
* `babilong_qa6`: Yes/No questions
* `babilong_qa7`: Counting
* `babilong_qa8`: Lists and sets
* `babilong_qa9`: Simple negation
* `babilong_qa10`: Indefinite knowledge
* `babilong_qa11`:
* `babilong_qa12`:
* `babilong_qa13`:
* `babilong_qa14`:
* `babilong_qa15`:
* `babilong_qa16`:
* `babilong_qa17`:
* `babilong_qa18`:
* `babilong_qa19`:
* `babilong_qa20`:

Evaluation sets are in 100 samples and 1000 samples per task and per length, at 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512k, 1M and 10M tokens. (Note: the 1k dataset only has up to 128k context lengths)
