# Benchmark Catalog and Metrics

This catalog records the tasks observed in the local MLPerf v6.0/v6.1 snapshots.
Always verify the task manifest in the repository for the round being run.

## Mobile

The v6.0 result table contains seven rows on a Samsung Galaxy S26:

- Image Classification v2 Online: MobileNetV4, SingleStream.
- Image Classification v2 Offline: MobileNetV4, Offline.
- Object Detection: MobileDETs, SingleStream.
- Image Segmentation v2: MOSAIC, SingleStream.
- Language Understanding: Mobile-BERT, SingleStream.
- Super Resolution: EDSR F32B5, SingleStream.
- Stable Diffusion: Stable Diffusion, SingleStream.

The result export uses queries per second for these rows. The mobile app also
exposes task-specific accuracy and latency. Report the exact export unit rather
than silently converting QPS to latency. Mobile results are intended for
physical phones and their actual CPU, GPU, or NPU execution path.

## Tiny

The v1.4 workbook has closed and open sheets. Its task columns are `ad`, `ic`,
`ic2`, `kws`, `sww`, and `vww`; model names include FC AutoEncoder,
ResNet-V1/V2, DSCNN, and MobileNetV1 (0.25x). Check the task manifest before
mapping a code to a human-readable task.

Tiny reports task performance and power-related measurements. The workbook uses
units such as milliseconds for latency and microjoules for energy, plus a
separate accuracy field. Latency, energy, and accuracy must remain separate.

## Inference Edge, Closed Snapshot

The v6.1 result export contains:

- YOLO-95 in SingleStream, MultiStream, and Offline.
- YOLO-99 in SingleStream, MultiStream, and Offline.
- Whisper in Offline.
- Agentic Edge in SingleStream.

## Inference Edge, Open Snapshot

The observed open list contains:

- ResNet-50 in SingleStream and MultiStream.
- Llama 3.1 8B Edge in SingleStream and Offline.
- Agentic Edge with Qwen3.6-27B in SingleStream.

## Inference Datacenter, Closed Snapshot

The v6.1 result export contains:

- DLRM-v3: Server and Offline.
- Llama 3.1 8B: Server, Offline, and Interactive.
- Llama 2 70B: Server, Offline, and Interactive.
- R-GAT: Offline.
- DeepSeek-R1: Server, Offline, and Interactive.
- Whisper: Offline.
- Text-to-Video: SingleStream and Offline.
- VLM: Server, Offline, and Interactive.
- GPT-OSS-120B: Server, Offline, and Interactive.
- End-to-End QA: Offline.
- End-to-End Vector DB: Offline.

## Inference Datacenter, Open Snapshot

The observed open list contains:

- DLRM-v3: Server and Offline.
- Llama 3.1 8B: Offline.
- Llama 2 70B: Offline.
- DeepSeek-R1: Server and Offline.
- VLM: Offline.

The Edge and Datacenter lists describe observed result rows, not a promise that
the same rows will exist in another release.

## Scenarios

- `SingleStream`: no request overlap; emphasizes per-query latency.
- `MultiStream`: concurrent queries in a fixed group; latency constraints apply.
- `Offline`: all available samples may be processed together; emphasizes
  throughput.
- `Server`: requests arrive under a modeled load; find the maximum throughput
  that still satisfies latency constraints.
- `Interactive`: token-generation throughput under time-to-first-token and
  time-per-output-token constraints.

Not every workload supports every scenario. Follow the task README rather than
inventing an omitted scenario.

## Open and Closed Division

Closed division fixes the model and requires both performance and accuracy
evidence, normally at 99% or 99.9% of reference accuracy depending on the task.
It also has compliance and system constraints. Open division relaxes model and
implementation constraints but still requires the current round's accuracy and
reporting rules.

For the checked repository snapshot:

- Datacenter closed requires Offline and Server, each at least ten minutes.
- Edge closed requires SingleStream, Offline, and task-applicable MultiStream,
  each at least ten minutes.
- Closed runs require additional compliance runs; Offline may require at least
  24,756 queries and can take much longer on weak hardware.
- Open submissions may use one scenario, with no closed compliance runs, but
  still need valid accuracy evidence unless the rules permit an exception.

## Accuracy and Performance

Accuracy is task-specific: top-1 accuracy, mAP, Dice, DICE/F1, ROUGE, CLIP/FID,
or another declared metric. Performance is also task-specific and can use
latency, queries/s, samples/s, tokens/s, or energy. Never infer accuracy from a
performance table that does not contain an accuracy field.
