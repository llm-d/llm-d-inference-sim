## Prefill/Decode Disaggregation Deployment Guide

This guide demonstrates how to deploy the LLM Disaggregation Simulator (llm-d-sim) in a Kubernetes cluster using a separated Prefill and Decode (P/D) architecture. 

The [`routing-sidecar`](https://github.com/llm-d/llm-d-routing-sidecar) runs alongside the Decode service and acts as a reverse proxy: it receives client requests, forwards the prefill phase to a dedicated Prefill service (based on the x-prefiller-host-port header), and then handles the decode phase locally.

> This is a standalone simulation setup, intended for testing and validating P/D workflows without requiring the [llm-d-inference-scheduler](https://github.com/llm-d/llm-d-inference-scheduler). 
> It uses standard Kubernetes Services for internal communication between components.

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
To access both the Decode services from your local machine, use `kubectl port-forward` to forward their ports to your localhost.

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
>This header must be provided by the client in standalone mode. It tells the `routing-sidecar` where to send the prefill request. The value should be a Kubernetes Service name + port (or any resolvable `host:port` reachable from the sidecar pod).
>
> In production deployments using `llm-d-inference-scheduler`, this header is typically injected automatically by the scheduler or gateway—but in this standalone simulator, the client must set it explicitly.


#### Realistic Config

This example already configures non-zero latency parameters to reflect real-world P/D disaggregation behavior:

```yaml
- "--prefill-time-per-token=200"   # ~200ms per input token for prefill computation
- "--prefill-time-std-dev=3"       # ±3ms jitter to simulate system noise
```

Parameter meanings:
- `prefill-time-per-token`: Average time (e.g., 100ms. in milliseconds if unit is missing) to process each prompt token during the prefill phase. Higher values emphasize the cost of large prompts.
- `prefill-time-std-dev`: Standard deviation (in ms) of prefill latency, introducing realistic variation across requests.