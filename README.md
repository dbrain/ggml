# ggml

[Manifesto](https://github.com/ggerganov/llama.cpp/discussions/205)

Tensor library for machine learning

***Note that this project is under active development. \
Some of the development is currently happening in the [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) repos***

## This fork

> ⚠️See warning / rant over here: https://github.com/dbrain/qwen3-tts.cpp#added-in-this-fork ⚠️
> 
> TL;DR Robot generated guff, technically a software engineer and I'm sure I could "work this out myself" if I wanted to, but in its currently state it's entirely "Claude take the wheel"
> 
> Background: I wanted to play with qwen3-tts and the current state was unusable on my hardware (RTX 3060 12GB). The changes here and at https://github.com/dbrain/qwen3-tts.cpp took me from 0.1 RTS -> 2.8 RTS on Q8, i.e. "would need to pre-process text" to "I can TTS at 2x" with no quality drops that I can perceive (but I may be tone deaf).
> 
> If I couldn't "hey claude, stop being a quitter and do that massive change you said might boost performance" on a loop I probably would have started looking at this and maybe even learnt to translate what the commit messages are saying to human (in between mma_syncing my gemms), but I definitely would have just bailed before getting to any kind of useful progress.
>
> Probably a "never upstream change" - I'm sure as hell not creating a PR for (likely horrible) code I may never bother to understand properly. Potentially a "Only works on my hardware". Basically "Here be dragons".

## Features

- Low-level cross-platform implementation
- Integer quantization support
- Broad hardware support
- Automatic differentiation
- ADAM and L-BFGS optimizers
- No third-party dependencies
- Zero memory allocations during runtime

## Build

```bash
git clone https://github.com/ggml-org/ggml
cd ggml

# install python dependencies in a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# build the examples
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
```

## GPT inference (example)

```bash
# run the GPT-2 small 117M model
../examples/gpt-2/download-ggml-model.sh 117M
./bin/gpt-2-backend -m models/gpt-2-117M/ggml-model.bin -p "This is an example"
```

For more information, checkout the corresponding programs in the [examples](examples) folder.

## Resources

- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [The GGUF file format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
