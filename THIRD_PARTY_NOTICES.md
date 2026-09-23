# Third-party notices

## Phoenix evaluator templates

The files in `src/jev/templates/` were copied from the generated Phoenix TypeScript evaluator templates at commit `6f03f903b8d3eddffe11e6b695c1c24e244f946a` in [Arize-ai/phoenix](https://github.com/Arize-ai/phoenix). Their SHA-256 digests are locked in `src/jev/templates/index.ts` and checked by the offline tests.

Phoenix is distributed under the Elastic License 2.0. See `LICENSE`.

## NVIDIA Nemotron-PII

`src/fixtures/pii_detection.nemotron.jsonl` is a 150-record sample of [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII). The sample preserves each source record's UID and metadata. NVIDIA publishes the dataset under the Creative Commons Attribution 4.0 International license.

Suggested citation:

```bibtex
@dataset{nemotron-pii,
  author = {Amy Steier and Andre Manoel and Alexa Haushalter and Maarten Van Segbroeck},
  title = {Nemotron-PII: Synthesized Data for Privacy-Preserving AI},
  year = {2025},
  publisher = {NVIDIA},
  url = {https://huggingface.co/datasets/nvidia/Nemotron-PII}
}
```
