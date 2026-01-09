## K8s Tutorial

This is the codebase for the EIDF Tutorial. It is a simple ML training job that trains some CNNs on CIFAR10.

Instructions:

1. Download docker desktop https://www.docker.com/products/docker-desktop/

2. Copy this repo.

3. If you want to test locally, created your own `.env` file and add your Wandb API Key `WANDB_API_KEY=<your_key>`

4. Make sure docker is up and running.

5. If you are on a non apple silicon system run:
    `docker build -f Dockerfile.simple -t <your_docker_username>/k8s_tutorial_simple:0.0.1 .`
   
   If you are on a mac, run:
    `docker buildx build --platform=linux/amd64 -f Dockerfile.simple -t <your_docker_username>/k8s_tutorial_simple:0.0.1 .`

6. Make sure you are logged into the docker cli `docker login`

7. Push to your docker hub `docker push <your_docker_username>/k8s_tutorial_simple:0.0.1`

8. SSH into your cluster, for me thats `ssh med_k8s`

9. Edit the `k8s/training_job_simple.yaml` file to include your credentials (mine wont work for your id).

10. Sync the k8s directory to the cluster `rsync -r k8s med_k8s:k8s_tutorial` (making sure the k8s_tutorial dir exists on the cluster already).

11. Submit your job with `kubectl create training_job_simple.yaml`

12. Run `kubectl get pods` and get your pods name.

13. Monitor the logs of the training job with `kubectl log follow <your_pod_name>`

14. Done :)


