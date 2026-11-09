# Phase 1.5 feature contract

- Numeric feature: each result row uses the scalar `total_accelerators` value.
- The sorted reported-count tuple is retained only as `accelerator_count_values` for audit grouping.
- The sensitivity key excludes the count tuple and uses `submitter|accelerator|processor`.
- The count tuple is a system-level report signature, not a numeric model feature.

| System | Reported counts | Sensitivity group |
| --- | --- | --- |
| `datacenter|HPE|HPE Compute XD690 (8x NVIDIA B300-SXM-270GB TensorRT)|datacenter` | `8;16` | `HPE|NVIDIA B300-SXM-270GB|Intel(R) Xeon(R) 6776P` |
| `datacenter|NVIDIA|NVIDIA GB300 NVL72 (72x GB300-288GB_aarch64 TensorRT)|datacenter` | `72;288` | `NVIDIA|NVIDIA GB300|NVIDIA Grace CPU` |
