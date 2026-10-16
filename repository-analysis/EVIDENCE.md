# Evidence and Review Record

## Scope and screening

**Full analysis:** an educational experiment with substantial custom CPU/CUDA implementations and a useful debugging history; not merely an unmodified tutorial. This does not establish independent invention of neural-network algorithms or recommend featuring the project.

Inspected all 31 tracked current files by inventory and the principal implementation areas: root README, CMake, configuration example, network and I/O modules, headers, and all four diagnostic source areas. Inspected the complete local commit listing and representative milestone diffs, including historical Python training code. Local refs are `main`, `origin/main`, and `origin/HEAD`, all at the analyzed commit; no local tags or additional branches. No remotes fetched. No existing analysis directory or human corrections were present. No AGENTS.md was found beneath the parent directory.

Analysis is static: no repository code, build scripts, training, tests, or dependency installation was executed. Dataset files, trained models, raw benchmark logs, and CI configuration are absent from tracked current files. No external research was necessary; dataset URLs are retained as existing README references, not independently availability-checked.

## Claim ledger

| Claim | Locally verifiable evidence | Confidence |
| --- | --- | --- |
| Educational digit classifier inspired by 3Blue1Brown | Root `README.md`; user recollection; `src/main.cpp` | High for documented purpose; tutorial implementation provenance is not established |
| Custom CPU and CUDA paths implement loading, normalization, initialization, forward/backward computation and evaluation | `src/io/io_utils.cpp`, `normalize.cpp`; `src/network/init.cpp`, `cpu.cpp`, `gpu.cu`, `train.cpp`; `src/main.cpp` | High: direct implementation evidence |
| Current shape is 784–512–256–10 for 28×28 data, with sigmoid forward activations | `src/main.cpp`; `forward_propagation_cpu/gpu`; `bb33a18` | High: input size derives from file dimensions; hidden/output sizes are fixed |
| Seeded comparison starts from copies of the same initialized network | `src/main.cpp` (`network_copy`); `src/network/init.cpp`; `0121c19` | High: both flags run separate copies; seed zero remains random |
| GPU network weights persist across training, but scratch allocations and host transfers remain | `allocate_network_data_on_gpu`, `forward_propagation_gpu`, `backpropagation_gpu`; `src/network/train.cpp`; `7d80d97` | High; root TODO claiming allocation only once overstates current behavior |
| Pointer-offset bug and stale last-image activation bug were distinct | `33fea1a` modifies historical `src/gpu.cu` to replace offset into delta with copied `next_delta`; `c5e3c98` modifies `src/network/gpu.cu` to copy host averages to GPU; root README debugging notes | High: diffs substantiate both corrections |
| Batch accumulation exists but does not implement ordinary per-example gradient averaging | `train_on_a_batch_cpu/gpu` in `src/network/train.cpp`; `backpropagation_cpu/gpu`; `d54fb9b`, `c5e3c98` | High: one delta vector per layer derives from batch averages and is reused across examples |
| October “Switch to relu” changed derivatives without changing sigmoid forward propagation | `aebbf25`; current `src/network/cpu.cpp`, `gpu.cu` | High: commit title is stronger than implementation |
| Root README reports 90.17% accuracy for both paths | `README.md`, `259068b`, `534f910`; comment in `src/main.cpp` | Medium for achieved historical result; high that it is recorded, no independent reproduction |
| Existing timing values are not a controlled benchmark | README reports GPU train/test 51.12/28.09 and CPU 625.72/42.56; timers in `src/network/train.cpp` include sleeps/logging and different allocation/transfer boundaries | High for recorded values and measurement limitations; no hardware/build/seed/run log supports generalized speedup |
| Diagnostic programs are manual rather than automated correctness tests | `CMakeLists.txt`; `src/test/matrix_mult.cpp`, `matrix_mult.cu`, `atomic.cu`, `cuda_inspection.cu`; no CTest registration | High: printed results lack assertions |
| Current substantive stack is C++17, CUDA, CMake, Boost.Program_options | `CMakeLists.txt`, `src/io/cml.cpp` | High; TensorFlow/NumPy were historical only |
| Early Python/Keras experiment was removed | `85e5f4f:python/train.py` uses TensorFlow/Keras and NumPy; `d43097c` removes Python directory | High; not represented as a current dependency or original ML framework |

## Timeline basis

- **Earliest observed activity:** July 28, 2024, `85e5f4f` (author and committer dates agree): IDX reader, Python experiment, initial build files.
- July 29–August 4: normalization (`f7d1d1d`), layer initialization (`796ef72`), CPU/GPU forward propagation (`93c2366`, `99301c5`).
- August 11–13: matrix multiplication diagnostics and corrections (`afce751`, `5a92e17`).
- August 20–29: CPU backpropagation (`0bfbb9d`), training loop (`62964ca`), CUDA training (`57d1a1c`), delta-buffer correction (`33fea1a`).
- September 2–3: persistent network allocations (`7d80d97`), module reorganization, deletion of Python code (`d43097c`); merge `8d29d81` references a development branch no longer separately available locally.
- September 8–29: CPU batches (`4a14496`), GPU batch feeding (`7a33677`), atomic accumulation (`17462c0`), double storage (`6afc707`), CPU accumulation (`d54fb9b`), averaged GPU activations (`c5e3c98`).
- **End of sustained development:** October 5, 2024. `aebbf25` changes derivatives, `bb33a18` enlarges the network, and `259068b` updates the README and training logging. No later source changes observed.
- **Latest observed activity:** October 13, 2026, `49248ac`; it and preceding `534f910` modify only README content after a roughly two-year gap.

There are 160 observed commits including a merge. July 28–October 5 spans 69 elapsed days (about ten weeks), consistent with “a couple of months.” Commits occur on every day of the week, so history does not support a weekends-only schedule. Commit timestamps do not establish working hours or effort; the recalled 8–10 hours per weekend remains unverified. Actual project start/end dates remain unknown; metadata records Git dates separately.

## Recollection cross-check and uncertainties

1. CPU/GPU side-by-side learning and the last-image batch bug are supported. The pointer recollection likely combines two documented incidents; it should not be repeated as a single proven root cause.
2. “First neural network,” CPU being easier, and C++/CUDA practice are personal context rather than independently verifiable repository facts. The early Keras experiment also belongs to this repository’s history.
3. Confirm whether author identities `vt156`, `Tumbleweed`, and `Vlad` all belong to the owner before relying on exclusive personal attribution. Multiple names and a merged pull request do not alone establish multiple people or sole authorship.
4. Confirm benchmark hardware, toolchain/build mode, seed, exact train/test counts, and commands. The reported 90.17% lines use learning rate 0.14 while a `src/main.cpp` comment says 1.4. Neither full reproducibility nor bitwise CPU/GPU equality is established.
5. No evidence establishes deployment, commercial use, production users, actual total effort, or a formal project completion date. Status is inactive based on source history and the owner’s recollection of stopping.

## Publication risks

- No obvious credentials were observed in the inspected source/configuration. This was not an exhaustive forensic scan of every historical blob. Do not publish personal author emails or local editor/toolchain paths from Git/editor metadata.
- README credits 3Blue1Brown and MNIST; preserve attribution. No license file was found, so reuse rights and third-party provenance should be reviewed before publication or redistribution of datasets.
- `EVIDENCE.md` and `CRITICISM.md` are intended for private review, but committing them to a public repository would expose them; website visibility flags do not make Git files private.
- Root README contains no displayed images, so there are no media references to preserve or broken local images to flag.
- Public descriptions qualify historical accuracy and do not repeat the README’s absolute CPU==GPU claim or infer production readiness.

### Date restoration confirmed by the owner

The owner confirmed that an October 2026 rebase renamed commit messages and changed committer dates; the development changes belong to 2024. At the owner's request, 14 historical commits' committer dates were restored to their existing author dates (September 28–October 5, 2024). October 5 is one week after September 28. Genuine 2026 README commits retain their 2026 dates. Commit hashes in this report were updated to the restored main history. A local `backup/before-date-restoration-20261016` branch preserves the pre-restoration history; remote-tracking refs were not changed. The rewritten main tip has an identical tree to the prior tip, and working files were left intact. The ref inventory earlier in this report describes the initial analysis, before this authorized rewrite.
