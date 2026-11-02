# Execution and Device Guidance

## Run Integrity

Start from the matching MLCommons suite and submission repository. Pin source
commit, container image, driver, runtime, model, dataset, compiler, precision,
and device configuration. Prefer official task READMEs and submission code over
an unverified script remembered from another round.

For a server, an isolated Docker environment helps reproduce software
dependencies, but host GPU drivers, firmware, topology, clocks, and cooling
remain part of the system. Record those outside the container too.

## Development Runs versus Submission Runs

Quick runs are appropriate for checking data loading, model export, QSL/SUT
integration, and result logging. They do not establish a valid MLPerf result.
Use the suite-supported duration or sample-count override, record it, and rerun
with the official workload before comparing results.

Do not inject artificial CPU load by default. The benchmark defines the test
environment; separate load is a new experiment and must be reported as such.
Follow the harness's warm-up, cooldown, and stabilization procedure, and keep
temperature, clock, and power telemetry separate from official logs.

Preserve the complete run directory. Depending on the implementation, this can
include `mlperf_log_summary.json`, `mlperf_log_detail.json`,
`mlperf_log_accuracy.json`, `mlperf_log_trace.json`, build metadata, compliance
outputs, and power logs. A LoadGen trace records request/response events; it is
not evidence of utilization or bandwidth.

## Physical iPhone with MLPerf Mobile

The physical-device path is:

1. Connect and unlock the iPhone 16 Pro Max, trust the computer, and enable
   **Settings > Privacy & Security > Developer Mode** if Xcode requests it.
2. Confirm the Mac sees the phone with `flutter devices` or
   `xcrun devicectl list devices`.
3. In the MLPerf app repository, build the native libraries with `make flutter`.
4. Open `flutter/ios/Runner.xcworkspace` in Xcode, enable automatic signing,
   select your team, and choose a globally unique bundle identifier.
5. Run a release build on the physical iPhone and select the installed app.

If Xcode shows no physical device, reconnect by cable, unlock, accept the trust
prompt, and restart Xcode. If iOS reports an untrusted developer, approve the
profile under **Settings > General > VPN & Device Management**.

A free Apple personal team normally produces a short-lived development profile;
it must be re-signed or reinstalled when it expires. A paid Apple Developer
Program membership supports longer-lived development certificates, but verify
the current annual price and terms with Apple. Firebase upload is optional
result collection or CI plumbing, not a requirement for a local benchmark.

Build information must be refreshed before saving a result:

```bash
make flutter/build-info
make flutter
cd flutter
flutter devices
flutter run -d <physical-device-id>
```

The exact Xcode, Flutter, Bazel, CocoaPods, protobuf, and Python versions matter;
use the versions pinned by the checked-out repository. Avoid the iOS Simulator
for reported performance: its compute path is not the phone's.

Large custom datasets can be placed under the app's **On My iPhone** folder.
The app may use a path such as `MLPerf/mlperf_datasets/...`; follow the current
custom-task documentation because the Files app can be slow or restrictive for
large directory trees.

## Tiny Devices

A development board includes an MCU, memory, flash, debugger, clocks, connectors,
and often sensors. “Board” and “MCU” are therefore not interchangeable. A
published board result does not prove that a similar MCU alone meets the task
memory, latency, energy, or toolchain requirements.

The local Tiny v1.4 directory does not list `FRDM-MCXN947`. The MCU pairs Cortex-M33
cores with NXP eIQ Neutron NPU, but “unpublished” does not imply competitive or
compliant. Establish fitness by building each candidate task, recording flash and
SRAM use, running on the physical board, and validating accuracy and latency.

For a CPU-only MCXN947 experiment, compile a TFLite Micro/CMSIS-NN or other
Cortex-M33 implementation without routing operators to eIQ Neutron. That produces
a defensible CPU baseline, not the board's NPU result. Keep the build flags and
runtime delegate configuration.

Raspberry Pi Pico with RP2040 provides dual Cortex-M0+ cores, 264 KB SRAM, and
no NPU. Expect substantial memory and compute constraints for Tiny workloads;
prove feasibility on the exact task rather than extrapolating from board specs.
Older Cortex-M4 boards such as FRDM-K64F/K66F are useful CPU-only controls but
are not equivalent to newer NPU-equipped systems.

## Ordering Evidence for a Candidate Board

Before recommending a board, collect the MCU core/clock/SRAM/flash, accelerator,
framework support, supported model operators, measured latency/energy, accuracy,
board availability, and exact submission status. If any required measurement is
missing, state that it remains unproven.
