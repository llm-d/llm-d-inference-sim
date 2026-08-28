# Get Started: CPU-only inference simulation

This folder reproduces the simulator smoke test we already ran with `llm-d-inference-sim`. It deploys four CPU-only simulator replicas behind one Kubernetes Service and sends an OpenAI-compatible chat-completions request through the Service.

The folder name is `get_started_qwen` because this work came from the Qwen/llm-d getting-started path. This smoke test does not load Qwen weights. It uses the dummy model name `test-sim-model`, so `llm-d-inference-sim` uses its simulated tokenizer and does not need a Hugging Face token, a render server, or a GPU.

## What this deploys

The default setup is the same one used in the initial experiment:

- Namespace: `llm-d-sim`
- Simulator image: `ghcr.io/llm-d/llm-d-inference-sim:v0.10.2`
- Replicas: `4`
- Model name: `test-sim-model`
- Response mode: `echo`
- Simulator port: `8000`
- GPU requests: none
- llm-d router/EPP: not used in this smoke test

The Kubernetes Service named `inference-sim` load-balances requests across the four simulator pods.

## Prerequisites

You need a working Kubernetes cluster and `kubectl` configured to access it. The cluster must be able to pull images from `ghcr.io`. The test script also requires `curl` on the machine where you run the script.

You can confirm Kubernetes access with:

```bash
kubectl cluster-info
kubectl get nodes
```

No NVIDIA runtime, GPU, Hugging Face token, or model download is required for this simulation.

## Run the simulation

From the repository root, switch to this branch and enter this directory:

```bash
git checkout get-started-qwen-sim
cd scripts/get_started_qwen
```

Deploy the namespace, four simulator pods, and Service:

```bash
./deploy.sh
```

The script waits for the Deployment to finish rolling out and then prints the pods. A healthy deployment should show four `Running` pods:

```text
NAME                             READY   STATUS
inference-sim-xxxxxxxxxx-xxxxx   1/1     Running
inference-sim-xxxxxxxxxx-xxxxx   1/1     Running
inference-sim-xxxxxxxxxx-xxxxx   1/1     Running
inference-sim-xxxxxxxxxx-xxxxx   1/1     Running
```

You can check them again with:

```bash
kubectl get pods -n llm-d-sim -o wide
```

## Send a test request

Run:

```bash
./test.sh
```

The script temporarily port-forwards the `inference-sim` Service to local port `8000`, sends this OpenAI-compatible request, prints the response, and closes the port-forward:

```json
{
  "model": "test-sim-model",
  "messages": [
    {
      "role": "user",
      "content": "Hello from llm-d simulation"
    }
  ],
  "max_tokens": 32
}
```

Because the simulator runs in `echo` mode, the returned content should reflect the input message.

You can change the local port or prompt without editing the script:

```bash
LOCAL_PORT=18000 PROMPT="test request" ./test.sh
```

## Inspect the deployment

Useful commands are:

```bash
kubectl get deployment,service,pods -n llm-d-sim
kubectl logs -n llm-d-sim deployment/inference-sim
kubectl describe deployment inference-sim -n llm-d-sim
```

The current manifest intentionally requests only CPU and memory resources. It does not request `nvidia.com/gpu`.

## Change the number of simulator backends

Edit `replicas` in `deployment.yaml`:

```yaml
spec:
  replicas: 4
```

Then run:

```bash
./deploy.sh
```

For example, setting `replicas: 8` creates eight simulated vLLM-compatible backends.

## Change simulator behavior

The simulator arguments are in `deployment.yaml`:

```yaml
args:
  - "--model"
  - "test-sim-model"
  - "--port"
  - "8000"
  - "--mode"
  - "echo"
```

This folder keeps the exact simple configuration used in the smoke test. Later experiments can add latency profiles, KV-cache simulation, load-dependent slowdown, failure injection, or a render server for real tokenization.

## Clean up

Delete the experiment namespace and everything created inside it with:

```bash
./cleanup.sh
```

You can use another namespace by setting `NAMESPACE` consistently:

```bash
NAMESPACE=my-sim ./deploy.sh
NAMESPACE=my-sim ./test.sh
NAMESPACE=my-sim ./cleanup.sh
```
