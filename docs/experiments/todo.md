# Pending Validation

## Versioned physical Parquet shards for distributed reads

- **Status:** design and benchmark pending.
- **Constraint:** preserve dense global modulo sample-index semantics across ranks;
  filter, resume, ordering, and stable-index behavior must remain unchanged.
- **Candidate design:** introduce a schema-versioned physical Parquet shard layout
  with an explicit global-index mapping, so ranks can decode disjoint row groups
  instead of redundantly decoding the same row group.
- **Design requirements:** define the index mapping, schema version, and an
  explicit migration path; do not add silent compatibility for an incompatible
  physical format.
- **Ready gate:** run a representative multi-rank benchmark that measures row-group
  decode count, I/O, CPU cost, and end-to-end throughput, and confirms repeated
  row-group decoding is a material bottleneck before implementing the redesign.
- **Acceptance:** reduce aggregate decode work or improve end-to-end throughput
  while preserving exact sample ownership and the existing filter/resume/index
  semantics.
