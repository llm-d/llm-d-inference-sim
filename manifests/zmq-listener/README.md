## ZMQ Listener
A Python-based ZeroMQ listener for capturing and processing KV cache events from LLM inference systems (vLLM, simulator, etc.).

### Overview
This component subscribes to ZMQ pub-sub events containing KV cache block operations. It receives and deserializes events about block storage, removal, and cache clearing operations, providing visibility into the distributed KV cache state.

### Architecture
The system consists of three components:

- ZMQ Listener: subscribes to KV cache events on a ZMQ socket
- Event Publisher(s): vLLM or simulator instances that publish KV cache events
- Kubernetes Service: headless service for the listener pod discovery

**Event Types**
- BlockStored: new blocks added to cache
- BlockRemoved: blocks evicted from cache
- AllBlocksCleared: full cache clear operation

**Message Format**

Messages follow a multi-part ZeroMQ format:
[topic, sequence_number, msgpack_payload]

Topic: Format kv@<pod_id>@<model_name> (e.g., kv@10.244.0.5@Qwen/Qwen3-0.6B)

Sequence: 64-bit big-endian sequence number

Payload: MessagePack-encoded EventBatch containing timestamped events

### Quick Start

**Build and Push Container Image**
Build multi-arch image
```bash
make zmq-image-build
```
Push to registry
```bash
make zmq-image-push
```

**Deploy to Kubernetes**
Set your namespace
```bash
export NAMESPACE=default
```

Deploy all components (listener + vLLM + simulator)
```bash
make deploy-zmq-all
```

Deploy individually
```bash
make deploy-zmq-listener
make deploy-vllm          
make deploy-sim 
```

### Configuration
**Environment Variables**
The listener accepts the following environment variable:

- ZMQ_PORT: the port to bind ZeroMQ socket (default: 5557)


**Kubernetes Configuration**
Edit manifests/zmq-listener/deploy_listener.yaml to customize:
```
- name: ZMQ_PORT
  value: "5557"  # Change port if needed

resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### vLLM Configuration
The vLLM deployment configures event publishing via:
```
--kv-events-config "{
  \"enable_kv_cache_events\":true,
  \"publisher\":\"zmq\",
  \"endpoint\":\"tcp://zmq-listener-service:5557\",
  \"topic\":\"kv@${POD_IP}@Qwen/Qwen3-0.6B\"
}"
```

### Clean up 
`delete-zmq-listener` Remove listener deployment <br>
`delete-sim` Remove simulator deployment <br>
`delete-vllm` Remove vLLM deployment <br>
`delete-zmq-all` Remove all deployments <br>
`clean-zmq` Delete all deployments and remove Docker image <br>
