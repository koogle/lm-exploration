## ClimbMix 124M: base vs math SFT

The same prompt and decoding settings run through five 124M models, one after another on
CPU.

- **Base 5.45B / 10.9B / 20B:** pretraining only (ClimbMix + code, warmup-stable-decay),
  stopped and decayed after that many tokens. Fluent text; arithmetic is mostly wrong
  (GSM8K and SVAMP about 1–2%).
- **Math SFT 5.45B / 20B:** those base models after one epoch of GSM8K + MetaMathQA
  worked solutions (GSM8K 15.8% / 20.4%, SVAMP 14.3% / 20.3%, zero-shot). They answer
  in steps ending with `#### N`, and are worse than their base models at general text and
  code.

Keep **Wrap as Question/Answer** on for math; turn it off for free text or code.
Temperature 0 is greedy and ignores the seed.
