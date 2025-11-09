## Prefill/Decode Disaggregation Deployment Guide

This guide demonstrates how to deploy the LLM Disaggregation Simulator (llm-d-sim) in a Kubernetes cluster using a separated Prefill and Decode (P/D) architecture. 
The [`routing-sidecar`](https://github.com/llm-d/llm-d-routing-sidecar) intelligently routes client requests to dedicated Prefill and Decode simulation services, enabling validation of disaggregated inference workflows.

### Quick Start

1. Deploy the Application
   Apply the provided manifest (e.g., vllm-sim-pd.yaml) to your Kubernetes cluster:

```bash
kubectl apply -f vllm-sim-pd.yaml
```

> This manifest defines two Deployments (vllm-sim-p for Prefill, vllm-sim-d for Decode) and two Services for internal and external communication.

2. Verify Pods Are Ready
   Check that all pods are running:

```bash
kubectl get pods -l 'llm-d.ai/role in (prefill,decode)'
```

Expected output:

```bash
NAME                          READY   STATUS    RESTARTS   AGE
vllm-sim-d-685b57d694-d6qxg   2/2     Running   0          12m
vllm-sim-p-7b768565d9-79j97   1/1     Running   0          12m
```

### Send a Disaggregated Request Using kubectl port-forward
To access both the Decode services from your local machine, use kubectl port-forward to forward their ports to your localhost.

### Forward the Decode Service Port
Open a terminal and run:

```bash
kubectl port-forward svc/vllm-sim-d-service 8000:8000
```

This command forwards port 8000 from the `vllm-sim-d-service` to your local machine's port 8000.

#### Test the Disaggregated Flow

Now, send a request to the forwarded Decode service port with the necessary headers:

```bash
curl -v http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-prefiller-host-port: vllm-sim-p-service:8000" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello from P/D architecture!"}],
    "max_tokens": 32
  }'
```

>  Critical Header:
>```
>x-prefiller-host-port: vllm-sim-p-service:8000
>```
>This header tells the sidecar where to send the prefill request. Since we have `vllm-sim-p-service:8000`, we specify it here.