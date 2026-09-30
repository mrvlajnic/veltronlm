# VeltronLM Status Report

Generated 2026-10-01 00:01:16  ·  version `0.1.0-alpha`  ·  commit `unavailable`

> Every number below was read from an artefact on disk. A measurement that was not taken is reported as missing rather than inferred.

## Compute

- `directml`: DirectML device, 8.0 GB, fp16=True, bf16=False
- `cpu`: CPU (4 threads), 31.96 GB, fp16=False, bf16=False

## Model registry

| Tier | Name | L | D | H | KV | F | V | Ctx | Parameters | Trainable here |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T0 | `nano` | 4 | 256 | 4 | 2 | 688 | 32,768 | 512 | 19,679,488 | yes |
| T1 | `micro` | 8 | 512 | 8 | 2 | 1376 | 32,768 | 1,024 | 55,715,328 | yes |
| T2 | `mini` | 16 | 1024 | 16 | 4 | 2752 | 32,768 | 2,048 | 244,354,048 | yes |
| T3 | `small` | 24 | 1536 | 12 | 4 | 4096 | 32,768 | 4,096 | 704,724,480 | no |
| T4 | `4b` | 36 | 3072 | 24 | 8 | 8192 | 65,536 | 8,192 | 4,026,765,312 | no |
| T4 | `4b-long` | 36 | 3072 | 24 | 8 | 8192 | 65,536 | 16,384 | 4,026,765,312 | no |
| T4 | `1b` | 24 | 2048 | 16 | 4 | 5632 | 65,536 | 8,192 | 1,350,672,384 | no |

## Pretraining

- **steps logged:** 28
- **latest step:** 700
- **tokens seen:** 11,468,800
- **train loss:** 5.4555 (EMA 5.3408)
- **throughput:** median 3,583 tok/s (range 296–3,681)
- **grad norm (latest):** 0.6486

| Step | Tokens | Val loss | Val perplexity |
|---:|---:|---:|---:|
| 250 | 11,468,800 | 5.7617 | 317.88 |
| 500 | 11,468,800 | 5.1015 | 164.27 |

## Dataset

- **version:** `dataset-v1` (manifest `272e134cf74c7367`)
- **documents:** 1,380 seen, 1,312 kept (95.1% accept)
- **characters kept:** 383,615,628
- **train tokens:** 19,485,297
- **val tokens:** 51,287
- **chars per token:** 3.593

**Filter funnel**

| Stage | Rejected |
|---|---:|
| pii_drop | 56 |
| dedup_near_lsh_band0 | 2 |
| dedup_near_lsh_band1 | 2 |
| dedup_near_lsh_band4 | 2 |
| too_short | 1 |
| dedup_near_lsh_band6 | 1 |
| dedup_near_lsh_band12 | 1 |
| dedup_near_lsh_band13 | 1 |
| dedup_near_lsh_band8 | 1 |
| bad_json_line | 1 |
| dedup_exact | 1 |

**Mixture**

| Category | Target | Achieved | Upsampled |
|---|---:|---:|---:|
| code | 0.220 | 0.220 | 165 |
| general | 0.340 | 0.340 | 377 |
| general_public_domain | 0.100 | 0.100 | 0 |
| sr | 0.120 | 0.120 | 138 |
| support | 0.100 | 0.100 | 116 |
| technical | 0.120 | 0.120 | 138 |

## Tokenizer

| Tokenizer | Vocab | Train time (s) | en_prose | sr_cyrillic | sr_latin | python | technical_markdown | json_structured |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `tok-mini-32k` | 32,768 | 2.6 | 3.646 | 2.819 | 2.177 | 3.910 | 3.696 | 3.416 |
| `tok-4b-64k` | 57,611 | 2.9 | 3.788 | 3.484 | 2.346 | 4.032 | 3.826 | 3.597 |

All round-trip failures: **0** across every slice.

## Retrieval

- **hit@1:** 1.0000
- **recall@5:** 1.0000

| Query | Expected | Top hit | Score |
|---|---|---|---|
| How long is the warranty? | policy-warranty | `policy-warranty` | ok |
| My hub shows as offline | trouble-hub-offline | `trouble-hub-offline` | ok |
| veltron x1 status light blinking amber pairi | product-veltron-hub, trouble-hub-offline | `product-veltron-hub` | ok |
| What is the turnaround time for an RMA? | policy-warranty | `policy-warranty` | ok |
| Koliko traje garancija? | policy-warranty | `policy-warranty` | ok |
| Can I get a refund after 45 days? | policy-refund | `policy-refund` | ok |
| Where is my parcel? tracking says nothing | policy-shipping | `policy-shipping` | ok |
| I forgot my password | policy-security | `policy-security` | ok |
| What are the dimensions of the VeltronSense? | product-veltron-sense | `product-veltron-sense` | ok |
| How much is the Plus plan? | product-veltron-cloud | `product-veltron-cloud` | ok |
| Someone logged into my account and I did not | policy-security | `policy-security` | ok |
| Can I close my account and delete my data? | faq-general, faq-privacy-serbian, policy-privacy-data | `policy-privacy-data` | ok |
| I want to send the device back for repair | policy-refund, policy-warranty | `policy-warranty` | ok |
| Does the app need Bluetooth permission to pa | trouble-app-pairing | `trouble-app-pairing` | ok |

## Checkpoints

**micro-pretrain**: 2 checkpoint(s) on disk

**timing-bs4**: 0 checkpoint(s) on disk


## Benchmarks

**Compute**

- accelerator peak: **1.5008 TFLOP/s**
- cpu peak: **0.2145 TFLOP/s**
- speedup: **7.0x**
- `veltronlm-nano` train step: 1,576 tok/s (1.3 s/step)
- `veltronlm-micro` train step: 160 tok/s (12.783 s/step)

**Inference**

- `micro`: 55,715,328 params, 0.104 GiB weights, 13.63 tok/s median
