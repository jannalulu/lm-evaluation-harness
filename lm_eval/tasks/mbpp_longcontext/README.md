# MBPP Long-Context

## Overview

MBPP Long-Context combines coding problems from the [MBPP dataset](https://github.com/google-research/google-research/tree/master/mbpp) with long-context distractors from [BABILong](https://github.com/booydar/babilong) to evaluate coding ability with long-context distractors.

### Tasks
This benchmark uses the 500 questions from the MBPP test set, prepended with long-context fields from babilong.

> [!NOTE]
> When using mbpp_longcontext tasks, please note:
> 1. The default maximum sequence length is 0k. For calculating metrics of different max seq lengths, specify additional lengths using the metadata parameter:
>   `--metadata '{"max_seq_lengths":"0k,1k,2k,4k,8k,16k,32k,64k,128k,196k,256k,512k,1M"}'`. The config currently only takes one context length at a time. The metadata parameter can also be passed to the TaskManager (metadata: dict).

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
