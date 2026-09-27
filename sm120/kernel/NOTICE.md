# Vendored mixfp4 kernel: provenance and license status

- **Source:** https://github.com/brian030128/mixfp4 at `7b3ab34ebc4a31b396a27fa6aea3650259714cf9`. The vendored
  files and their upstream hashes are in `VENDORED.json`; the local, compile-time-only changes are in
  `LOCAL_CHANGES.md` / `LOCAL_CHANGES.patch`.
- **License status:** upstream has no license file. The kernel is vendored here with the author's permission. The
  author is a member of the same research team, as confirmed by the user on 2026-09-27.
- **Before any public release,** an explicit license must be added for this code; the team decides which. No
  license text has been chosen or added here.
- **Third-party code inside:** `src/collective/sm120_blockscaled_mma_tma_mixed.hpp` is derived from CUTLASS and keeps
  its BSD-3-Clause header. CUTLASS itself is the submodule `sm120/third_party/cutlass`, which carries its own
  `LICENSE.txt`.
