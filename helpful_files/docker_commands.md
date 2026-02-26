# Basic Docker Commands

## Pull an Image
```bash
# Pull from Docker Hub
docker pull <image-name>

# Pull specific version
docker pull <image-name>:<tag>
```

## Build an Image
```bash
# Build from Dockerfile in current directory
docker build -t <image-name> .

# Build with specific tag
docker build -t <image-name>:<tag> .

# If you are using a Mac and need to cross-platform build.
docker buildx build -t <image-name>:<tag> .
```

## Run a Container
```bash
# Run container
docker run <image-name>

# Run in background (detached)
docker run -d <image-name>
```

## Tag an Image
```bash
# Tag for registry
docker tag <image-name>:<tag> <registry>/<image-name>:<tag>

# Example for Docker Hub
docker tag my-app:latest username/my-app:latest

# Example for private registry
docker tag my-app:latest registry.example.com/my-app:latest
```

## Push to Registry
```bash
# Login first
docker login

# Push to Docker Hub
docker push username/<image-name>:<tag>

# Push to private registry
docker push registry.example.com/<image-name>:<tag>
```

## Basic Workflow Example
```bash
# 1. Build your image
docker build -t my-app:latest .

# 2. Tag for registry
docker tag my-app:latest username/my-app:latest

# 3. Push to registry
docker push username/my-app:latest

# 4. Pull on another machine
docker pull username/my-app:latest

# 5. Run the container
docker run username/my-app:latest
```
