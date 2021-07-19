# Dockerfile

```bash 
docker build -t ctpet .
docker build -t pix2pix .
```

## Tensorboard

```bash 
tensorboard --logdir=runs --port=8088
```

## Training 
```bash 
CUDA_VISIBLE_DEVICES=0 python Train_AE_2D.py
```

## Testing 
```bash 

```

## Testing on NIFTI files (3D)

```bash 
```
## License

## Docker (personal)
```bash 
docker run -it --rm --gpus all --shm-size=50G --user $(id -u):$(id -g) --cpuset-cpus=0-9 \
-v /rsrch1/ip/msalehjahromi/Codes/MoriRichard:/home/msalehjahromi/MoriRichard \
-v /rsrch1/ip/msalehjahromi/data:/Data \
--name MR ctpet:latest

```



docker run -it --rm --gpus all --shm-size=50G --user $(id -u):$(id -g) --cpuset-cpus=0-9 \
-v /rsrch1/ip/msalehjahromi/Codes/MoriRichard:/home/msalehjahromi/MoriRichard \
-v /rsrch1/ip/msalehjahromi/data:/Data \
pix2pix:latest

0-6
7-13
14-20
21-27