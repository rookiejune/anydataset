from __future__ import annotations

"""Internal IO helpers shared by store and filter implementations.

Modules in this package own filesystem atomicity, parquet primitives, and
count-bounded shard buffering. They do not define dataset, store, or filter
schemas.
"""
