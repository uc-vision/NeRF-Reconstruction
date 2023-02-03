# NeRF-Reconstruction

Goal of this repo is to have a stable NeRF implementation we can use as part of a pipeline.

At the time of writing reconstructing a scan looks like it is working (todo: some more correctness testing). From now on a development branch will be used for futher work, with the goal of keeping the main branch as stable as possible.

To run:
`python -m nerf.train`

Parameters are located at `./nerf/configs/config.yaml`. This includes everything from dataloading, network hyperparameters, training, logging, etc...

As some parts are more clear, and more relavant than others the documentation will be on a 'pull' basis for now (let me know if there is a piece of information you need, and I'll add it).

## TODO:
 - Minimal working implementation [x]
 - Correctness testing
 - Spec an API