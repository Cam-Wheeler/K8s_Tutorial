##  Tutorial

This is the codebase for the EIDF Tutorial. It is a simple ML training job that trains some CNNs on CIFAR10. Follow the steps below to get it up and running.

### Setup

Download docker desktop https://www.docker.com/products/docker-desktop/

Clone this repo.

Go to wandb and grab your api key: https://wandb.ai/site/

Either create your own `.env` file and add your Wandb API Key `WANDB_API_KEY=<your_key>` or set the environment variable `export WANDB_API_KEY=<your_key>`

Create a venv and install the requirements (use your fav manager for this). Make sure your env is activated.

Go to the conf directory and setup your configs. Just alter the `tutorial_dataset.yaml` and `tutorial_trainer.yaml` to point to the right place and list the correct device to run training on.

#### Run Locally

*   Run `python3 main.py dataset_conf=tutorial_dataset trainer_conf=tutorial_trainer`, the script should download CIFAR10 and train for 10 epochs. Check Wandb is logging correctly.

#### Build Docker Image

Make sure docker is up and running. There should be a docker emblem in your dock.

If you are on a non Apple Silicon system run:

```
docker build -f Dockerfile.simple -t <your_docker_username>/k8s_tutorial_simple:0.0.1 .
```

*   If you are on a Mac with Apple Silicon, run:

```
docker buildx build --platform=linux/amd64 -f Dockerfile.simple -t <your_docker_username>/k8s_tutorial_simple:0.0.1 .
```

*   The image will take a while to build on your first run!

#### Run Docker Locally (Optional)

Once your image is built, let's run it locally.

**Note:** If you don't have the NVIDIA Container Toolkit installed and want to test locally, you'll need to rebuild the image with the device set to `cpu` in your trainer config.

Run the following:

```
docker run \
  --volume <path to save checkpoints>:<path to save checkpoints> \
  --env-file <path to env file if you made one> \
  <your docker image name> \
  python3 main.py dataset_conf=tutorial_dataset trainer_conf=tutorial_trainer
```

If you set up an environment variable instead of a `.env` file, use this instead:

```
docker run \
  --volume <path to save checkpoints>:<path to save checkpoints> \
  --env WANDB_API_KEY=<your key here> \
  <your docker image name> \
  python3 main.py dataset_conf=tutorial_dataset trainer_conf=tutorial_trainer
```

For example, mine looks like this:

```
docker run \
  --volume /Users/cameronwheeler/code/k8s_tutorial/checkpoints/:/Users/cameronwheeler/code/k8s_tutorial/checkpoints/ \
  --env-file /Users/cameronwheeler/code/k8s_tutorial/.env \
  camwheeler135/k8s_tutorial_uv:0.0.3 \
  uv run main.py dataset_conf=local trainer_conf=docker_local_trainer
```

*   If you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed, you can use the `--gpus` flag to speed up training:

```
docker run \
  --gpus all \
  --volume <path to save checkpoints>:<path to save checkpoints> \
  --env-file <path to env file if you made one> \
  <your docker image name> \
  python3 main.py dataset_conf=tutorial_dataset trainer_conf=tutorial_trainer
```

*   Training should run inside the docker container this time! Again, check Wandb to ensure it's logging correctly.

#### Push To Docker Hub

Make sure you are logged into the docker cli `docker login`

Push to your docker hub `docker push <your_docker_username>/k8s_tutorial_simple:0.0.1`

### Data Management

We recommend that you store all persistent data in the PVC.

```
# Create personal directory in the mounted directory from VM: 
mkdir /home/eidf231/eidf231/shared/<your_username>
cd /home/eidf231/eidf231/shared/<your_username>

# Clone git repository to the created directory for demo experiment: 
git clone https://github.com/Cam-Wheeler/K8s_Tutorial.git
cd K8s_Tutorial

# Fill in your WANDB API key in .env
echo "WANDB_API_KEY=<your_API_key>" > .env

# Edit trainer config (e.g. wandb entity name) to match yours in conf/trainer_conf/hpc_cluster_trainer.yaml
wandb_entity_name: <your_entity_name> # replace <your_entity_name> with your entity name
```

### Let's Run On EIDF

SSH into EIDF via VSC Remote Explorer. You can also choose to use command line:

```
ssh -J <your_username>@eidf-gateway.epcc.ed.ac.uk <your_username>@10.24.8.217
```

Replace `<your_username>` in `k8s/tutorial_job.yaml` to your actual username. (e.g. `clee_ai4bi`)

```

# Submit the Kubernetes job
kubectl --namespace eidf231ns create -f k8s/tutorial_job.yaml

# Check that your pod is created and get its name
kubectl --namespace eidf231ns get pods

# Stream logs from the training job
kubectl --namespace eidf231ns logs -f <your_pod_name>
```

Done :)