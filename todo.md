# TODO

## Filter follow-ups

- Consider an advanced metrics sink interface only if users need outputs beyond
  the built-in parquet side cache.
- Consider optional columnar metrics schemas after the JSON payload shape has
  stabilized in real use.
- Revisit lazy partition index loading only together with cache snapshot
  lifecycle; current filtered datasets materialize selected indices eagerly.
