# Third-Party Notices

This file records third-party source files and dynamically loaded runtime shared libraries used in this repository.

| Project | Repository URL | Version / Commit | License | Local path | Hygon modifications |
| --- | --- | --- | --- | --- | --- |
| CUTLASS FA | http://42.228.13.241:10068/dcutoolkit/deeplearing/cutlass_3.2.1 | hytlass_dev / ee0f79a214a7a226ddd085f07f1c33948c2f8b28 | BSD-3-Clause | jax/_src/cudnn/cutlass_fa_attention_stablehlo.py |  |
| MIOpen | http://42.228.13.241:10068/dcutoolkit/deeplearing/miopen | develop / 64a77371abb957b6560b56349709923f8e74d416 | MIT | jax/_src/cudnn/miopen_attention_stablehlo.py | runtime library provided by DTK |
| jaxlib tools bundled licenses | https://github.com/jax-ml/jax.git | jax-v0.10.0 / a33ed614c58ee8a10d0b7536c50c2609c38500c1 | See jaxlib/tools/LICENSE.txt; GPL-3.0 and MPL-2.0 detected, legal/compliance approval required | jaxlib/tools/LICENSE.txt | license notice file copied from upstream JAX; no Hygon source modification |
