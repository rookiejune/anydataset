# Provider Services

`ProviderServer` keeps one provider or filter predicate in a dedicated process.
Materializer and filter workers communicate with that process through
`RemoteProviderFactory` or `RemoteFilterFactory` instead of loading the model
themselves.

The public boundary is:

| API | Responsibility |
| --- | --- |
| `ProviderServer` | Start a process, construct one provider on one device, serve calls, and stop the process. |
| `RemoteProviderFactory` | Route a materializer device string to a server and create a view-provider proxy. |
| `RemoteFilterFactory` | Route the current filter device to a server and create a predicate proxy. |
| `Runtime` | Configure process start methods and declare whether device state is local or server-owned. |

`Runtime(server_start_method=...)` does not create, start, discover, or stop a
server. The caller must manage every `ProviderServer` explicitly. A non-`None`
`server_start_method` tells the materializer or filter that the remote server
owns device state, so the local scan process does not set the torch device. It
also makes `reader_start_method="auto"` and `writer_start_method="auto"`
resolve to `fork`; without a server declaration they resolve to `spawn`.
Explicit reader or writer settings override `auto`.

Built-in background writers currently use threads by default, so
`writer_start_method` matters only when a process-backed writer is selected.

## Process Topology

The server process and the dataset-wide outer workers are separate concerns:

- `ProviderServer.start_method` starts the server process. Its default is
  `spawn`.
- One resolved materializer or filter device runs in the calling process; it
  does not start an outer device worker.
- Multiple resolved devices start one outer worker per device using
  `Runtime.process_start_method`, which defaults to `spawn`.
- `num_workers` controls the DataLoader readers inside the calling process or
  each outer device worker. With a server declaration, their `auto` start
  method is `fork`.

Set `Runtime.server_start_method` to the method used by the separately managed
server. The value records the topology; it is not a server launcher.

## Remote Materializer

This example keeps a LongCat provider on `cuda:0` while two forked DataLoader
workers read a canonical map-style store. Because only one materializer device
is selected, the materializer itself remains in the calling process.

```python
from __future__ import annotations

import os
from pathlib import Path

from anydataset import AnyDataset, Source, Spec
from anydataset.provider import LongCatProvider
from anydataset.provider_service import ProviderServer, RemoteProviderFactory
from anydataset.runtime import Runtime
from anydataset.store import ViewMaterializer
from anydataset.types import AudioView


BASE_STORE = Path("/data/audio")
DEVICE = "cuda:0"
AUTHKEY = b"replace-this-local-example-key"


def dataset_factory() -> AnyDataset:
    return AnyDataset(Spec(source=Source.STORE, path=str(BASE_STORE)))


def longcat_factory(device: str) -> LongCatProvider:
    return LongCatProvider(device=device)


def main() -> None:
    address = Path("/tmp") / f"anydataset-longcat-{os.getpid()}.sock"
    server = ProviderServer(
        address=address,
        provider_factory=longcat_factory,
        device=DEVICE,
        authkey=AUTHKEY,
        start_method="spawn",
    )

    with server:
        ViewMaterializer(
            "/data/audio-longcat",
            batch_size=8,
            num_workers=2,
            runtime=Runtime(server_start_method=server.start_method),
        ).write(
            dataset_factory=dataset_factory,
            provider_factory=RemoteProviderFactory(
                output=AudioView.LONGCAT,
                addresses={DEVICE: address},
                authkey=AUTHKEY,
            ),
            devices=DEVICE,
        )


if __name__ == "__main__":
    main()
```

`RemoteProviderFactory.output` is the local materializer contract. It must
match the output view produced by the provider in the server. Device keys in
`addresses` must exactly match the strings passed through `devices`.

## Remote Filter

Filters require a map-style dataset. This example assumes WMT19 has already
been materialized as a canonical store, then keeps the translation predicate in
its own CPU process while one forked DataLoader worker reads the store.

```python
from __future__ import annotations

import os
from pathlib import Path

from anydataset import AnyDataset, FilterRule, Preset, Source, Spec
from anydataset.provider_service import (
    ProviderServer,
    RemoteFilterFactory,
)
from anydataset.quality.translation import Predicate
from anydataset.runtime import Runtime


BASE_STORE = Path("/data/wmt19-zh-en")
DEVICE = "cpu"
AUTHKEY = b"replace-this-local-example-key"


def dataset_factory() -> AnyDataset:
    return AnyDataset(Spec(source=Source.STORE, path=str(BASE_STORE)))


def translation_factory(_device: str) -> Predicate:
    return Predicate.from_preset(
        Preset.WMT19,
        source_lang="zh",
        target_lang="en",
    )


def main() -> None:
    address = Path("/tmp") / f"anydataset-filter-{os.getpid()}.sock"
    server = ProviderServer(
        address=address,
        provider_factory=translation_factory,
        device=DEVICE,
        authkey=AUTHKEY,
        start_method="spawn",
    )

    with server:
        filtered = FilterRule(
            "wmt19-quality-v1-zh-en",
            RemoteFilterFactory(
                addresses={DEVICE: address},
                authkey=AUTHKEY,
            ),
        ).apply(
            dataset_factory=dataset_factory,
            device=DEVICE,
            batch_size=16,
            num_workers=1,
            metrics=True,
            runtime=Runtime(server_start_method=server.start_method),
        )

    print(filtered.counts)


if __name__ == "__main__":
    main()
```

`RemoteFilterFactory` is a zero-argument filter factory. Managed filter workers
set `ANYDATASET_FILTER_DEVICE`, which the factory uses to select an address. A
single-address factory also works when that environment variable is absent. A
multi-address factory used outside the managed filter environment must set the
device variable explicitly or it raises an error.

## Routing And Authentication

`ProviderServer.address` accepts a string, a `Path`, or a `(host, port)` tuple.
Use a unique Unix socket path for each live local server and ensure its parent
directory exists. String socket paths are removed before the server binds and
again when it exits. A TCP address can be used when filesystem sockets are not
appropriate.

`authkey` is optional bytes. When set, the same value must be passed to the
server and every remote factory; a mismatch raises
`multiprocessing.AuthenticationError`. Authentication does not encrypt IPC
payloads, so expose TCP listeners only on a trusted network or add an external
secure transport boundary.

For multiple devices, start one server per device and provide every exact
device-to-address mapping to the remote factory. `RemoteProviderFactory`
receives the device directly. `RemoteFilterFactory` obtains it from
`ANYDATASET_FILTER_DEVICE`.

## Serialization Boundary

The following values must be picklable:

- `ProviderServer.provider_factory` when the server uses `spawn`.
- Dataset, provider, and predicate factories sent to multi-device outer workers.
- Every IPC request and response, including canonical `Sample` mappings,
  provider view mappings, `Batch` objects, filter decisions, and provider
  outputs.

Define factories as module-level functions or callable classes. Do not use
lambdas, nested functions, open file handles, or process-local model objects as
spawned factory state. Construct the heavyweight model inside the server's
provider factory.

The service handles one request per connection. It does not provide persistent
connections, retries, shared-memory transfer, or an exactly-once side-effect
protocol. Providers and predicates should therefore be deterministic for a
given request, or own any retry-sensitive state explicitly.

## Lifecycle And Errors

Use `ProviderServer` as a context manager when possible. `start()` spawns the
process and waits for a successful readiness ping until `startup_timeout`.
`stop()` sends the close command, waits up to `shutdown_timeout`, and terminates
the process if it does not exit. Calling `start()` twice is an error; calling
`stop()` on a stopped server is harmless.

The server constructs one provider instance and reuses it until shutdown.
Calling `RemoteProvider.close()` or `RemoteFilterPredicate.close()` sends the
same service-wide close command, so do not call proxy `close()` when a
`ProviderServer` context owns the lifecycle.

Provider and predicate exceptions are returned as `RemoteProviderError` with
the remote exception type, message, and traceback. The server remains available
after an ordinary request failure. Address, connection, EOF, and authentication
failures remain transport exceptions. Pickling failures can occur locally while
sending a request or remotely while sending a result; there is no automatic
retry or fallback to a local provider.
