Thesis in Large Language Model (LLM) powered robotic agent

Using OpenPI published by Physical Intelligence team.

This project uses the π₀.₅ model, an upgraded version of π₀ with better open-world generalization trained with knowledge insulation.

Thesis will be using simulator provided by OpenPI and also Franka Panda provided by Örebro University.

Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring fsdp_devices in the training config. Please also note that the current training script does not yet support multi-node training.

Mode Memory Required Example GPU Inference > 8 GB RTX 4090 Fine-Tuning (LoRA) > 22.5 GB RTX 4090 Fine-Tuning (Full) > 70 GB A100 (80GB) / H100

This project is using only Inference and therefore > 8 GB is enough.

The repo has been tested with Ubuntu 22.04, and cannot assure that other operating systems will work.

All nesscary files is in examples/libero.

This project is testing 3 different methods. 

    VLA: By using the provided VLA by OpenPI. 
    
    VLM(OpenAI)+VLA: By using the provided VLA by OpenPI and a VLM from OpenAI with API. API-key is not needed, please do not abuse it (it can be removed later, please ask owner). API key exists on tsv-desktop

    VLM(Ollama)+VLA: By using the provided VLA by OpenPI and a VLM from Qwen. It have been tested with 7b parameters, however it will be changed later to 8b without CoT. 
When one of these methods wants to be tested, please go to first: examples/libero/Dockerfile.

Based on which method you want to use.. 
    
    #This is for when running with OpenAI
    #CMD ["/bin/bash", "-c", "source /.venv/bin/activate && python examples/libero/VLM_OpenAI.py $CLIENT_ARGS"]

    #This is the old cmd line for running the simulation without any VLM
    #CMD ["/bin/bash", "-c", "source /.venv/bin/activate && python examples/libero/main.py $CLIENT_ARGS"]

    #This is when running with Ollama locally
    #CMD ["/bin/bash", "-c", "source /.venv/bin/activate && python examples/libero/VLM_Ollama.py $CLIENT_ARGS"]

This is important. 

Don't forget to use Docker. 

To start VLA model: docker compose -f scripts/docker/compose.yml up
