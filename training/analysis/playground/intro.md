## ClimbMix 124M playground

The same prompt and decoding settings run through the selected 124M models, one after
another on CPU. The table lists every model, what training it had and its benchmark
scores; tick the models to compare.

**Prompt format** wraps your text the way the fine-tuned models were trained:

- **Math question:** `Question: …\nAnswer:`. Math models answer in steps ending with
  `#### N`.
- **Code task:** `Question: …\nAnswer:` and a new line. Describe the problem and name the
  function, for example ``Write a Python function `def add(a, b):`.``; code models
  answer with the function. Turn off **Stop at a blank line** for code.
- **Plain text:** your text as is, for base models.

Temperature 0 is greedy and ignores the seed.
