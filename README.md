# speech_tokenizer

A modified copy of [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)
(Zhang et al., *SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models*, ICLR 2024),
packaged as `speech_tokenizer` for the vocal intensity conversion code. Based on upstream commit
`30c96fb` (2024-06-09). Licensed under Apache-2.0, like the original: see `LICENSE`.

## Changes from upstream

- `SpeechTokenizer.from_pretrained` takes the configuration as a `dict` instead of a JSON path,
  and loads the checkpoint with `strict=False`.
- The single semantic projection of the first quantiser layer is replaced by one optional
  projection per layer (`codes_transform_dimensions`), and `forward` returns the list of
  projected codes.
- The residual vector quantiser detaches the residual between layers, and returns the
  per-layer codes under clearer names.
- The trainer is no longer imported by the package.

Each modified file starts with a notice saying so.

## Install

```bash
pip install "speech_tokenizer @ git+https://github.com/vers-project/speech_tokenizer@v0.1.0"
```
