# Running XCL on macOS (Apple Silicon, CrossOver)

This branch runs XCL on an Apple Silicon Mac: the Windows build of `xcl.exe` under CrossOver's
Wine, with Apple's D3DMetal as the D3D12 host and Rosetta 2 running the x86-64 code. It is a
port in progress on top of upstream `956a476`, not a supported platform.

## State

Tested on one machine: MacBook Pro (M5 Max, 64 GB), macOS 27.3, CrossOver 26.3.

- The newer retail game from the README table starts, reaches a steady 30 fps and shows its
  account prompt and first-run menus with complete text. An intro video fix exists on a
  separate branch.
- After the legal-terms screen the title queues tens of thousands of pipelines and the process
  grows to about 38 GB. Upstream's lazy pipeline building should fix that; this branch predates it.
- Stencil is lost: depth + stencil buffers are created depth-only (see below).
- Some pipelines fail to compile in D3DMetal ("Failed to compile fragment function") and are skipped.

## What the port changes

- **Host objects.** D3DMetal reads type information in front of an object's vtable and calls
  virtuals past the public interface. XCL's extended vtables mirror 16 words in front of the
  table and guard every upper slot on such a host (`d3d12x/real.rs`, `class_for`).
- **Unified memory.** D3DMetal has no `OpenExistingHeapFromAddress`. The committed window gets
  a copy in video memory: CPU writes are uploaded before each submit, GPU copies come back
  after it (`XCL_UMA_SHADOW=0` turns it off).
- **Memory the GPU cannot see.** Vertex and index data in ordinary memory is copied to upload
  buffers at bind and at execute. Title textures in memory that is not write-watched are
  compared by content and uploaded again when they changed.
- **Depth formats.** Metal makes no depth-only view of a depth + stencil texture and aborts
  the process instead of returning an error. On a native host every depth + stencil family
  becomes the depth-only R32 family (`XCL_DEPTH_STENCIL=1` restores the Windows mapping).
- **Instructions Rosetta 2 does not run:** INSERTQ, CLZERO, MONITORX/MWAITX, ADX, RDSEED,
  MOVNTSS/SD and CLWB are emulated or redirected (`isa.rs`, `eh.rs`).
- **The title's stall watchdog** fires when the host is slow for a moment; its deliberate
  crash is skipped (`XCL_TITLE_WATCHDOG=1` keeps it).
- **Network is off by default** (`XCL_NETWORK=1` turns it on). A retail title's crash
  reporter otherwise uploads minidumps.
- **Running unattended:** `XCL_BACKGROUND=1` opens the window without taking the focus and
  never holds the cursor; `XCL_REMOTE_INPUT=<file>` presses the buttons written to that file.

## Build

```sh
brew install mingw-w64
rustup target add x86_64-pc-windows-gnu
cargo build --release -p xcl --target x86_64-pc-windows-gnu
```

Put `dxcompiler.dll` and `dxil.dll` (from the DirectX Shader Compiler release) next to
`target/x86_64-pc-windows-gnu/release/xcl.exe`.

## Run

Create a Windows 10 64-bit bottle in CrossOver with D3DMetal as its graphics backend, then:

```sh
CX="$HOME/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin"
XCL_BACKGROUND=1 XCL_DATA_ROOT="Z:/path/to/data" WINEDEBUG=-all \
  "$CX/wine" --bottle <bottle> "Z:/path/to/xcl.exe" run "Z:/path/to/game" < /dev/null
```

Keep `XCL_DATA_ROOT` outside the game folder. Stdin from `/dev/null` matters: on an error XCL
waits for Enter. A game instance takes 20 to 40 GB here, so one at a time.

Useful switches: `XCL_FRAME_DUMP=n` (every n-th frame as a BMP in the data folder's `logs/`),
`XCL_PRESS_A_AT=130,145` (seconds), `XCL_DUMP_THREADS_AT=secs`, `XCL_TRACE_DEPTH_VIEWS=1`.
