from ast import Or
import zmq
import os
import sys
import logging
import msgpack
import signal
import struct
import time
from typing import Any, List, Optional, Union, Dict
from dataclasses import dataclass, field

# Configure logging to stdout so kubectl logs can capture it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration constants
POLL_TIMEOUT_MS = 250
EXPECTED_MESSAGE_PARTS = 3

# Retry configuration
MAX_BIND_RETRIES = 30
RETRY_BACKOFF_BASE = 2

# Event types
BLOCK_STORED_TAG = "BlockStored"
BLOCK_REMOVED_TAG = "BlockRemoved" 
ALL_BLOCKS_CLEARED_TAG = "AllBlocksCleared"

@dataclass
class EventBatch:
    ts: float
    events: List[bytes]
    data_parallel_rank: Optional[int] = None

@dataclass
class BlockStored:
    block_hashes: List[Union[int, bytes]]
    parent_block_hash: Optional[Union[int, bytes]] = None
    token_ids: List[int] = field(default_factory=list)
    block_size: int = 0
    lora_id: Optional[int] = None
    medium: Optional[str] = None
    lora_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': BLOCK_STORED_TAG,
            'block_hashes': self.block_hashes,
            'parent_block_hash': self.parent_block_hash,
            'token_ids': self.token_ids,
            'block_size': self.block_size,
            'lora_id': self.lora_id,
            'medium': self.medium,
            'lora_name': self.lora_name
        }

    @classmethod
    def from_msgpack(cls, event: List[Any]) -> 'BlockStored':
        return cls(
            block_hashes=get_event_field(event, 0, []),
            parent_block_hash=get_event_field(event, 1, None),
            token_ids=get_event_field(event, 2, []),
            block_size=get_event_field(event, 3, 0),
            lora_id=get_event_field(event, 4, None),
            medium=get_event_field(event, 5, None),
            lora_name=get_event_field(event, 6, None),
        )
    
@dataclass
class BlockRemoved:
    block_hashes: List[Union[int, bytes]]
    medium: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': BLOCK_REMOVED_TAG, 
            'block_hashes': self.block_hashes, 
            'medium': self.medium
        }
    
    @classmethod
    def from_msgpack(cls, event: List[Any]) -> 'BlockRemoved':
        return cls(
            block_hashes=get_event_field(event, 0, []),
            medium=get_event_field(event, 1, None),
        )

@dataclass  
class AllBlocksCleared:
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': ALL_BLOCKS_CLEARED_TAG
        }
    
    @classmethod
    def from_msgpack(cls, event: List[Any]) -> 'AllBlocksCleared':
        return cls()
    
EventType = Union[BlockStored, BlockRemoved, AllBlocksCleared]

def get_event_field(event: List[Any], index: int, default_value: Any) -> Any:
    return event[index] if index < len(event) else default_value
    
def unmarshal_kv_event(raw_event: bytes) -> tuple[EventType|None, str]:
    """Unmarshal raw msgpack event into typed event, matching Go UnmarshalKVEvent logic"""
    try:
        # Parse as tagged union: first element is tag string, rest is payload
        tagged_union = msgpack.unpackb(raw_event, raw=False)        
        if not isinstance(tagged_union, list) or len(tagged_union) < 1:
            return None, "malformed tagged union: no tag"
        
        tag = tagged_union[0]
        if not isinstance(tag, str):
            return None, f"invalid tag type: {type(tag)}"
        
        # Re-marshal payload as single msgpack bytes (matches Go logic)
        payload_bytes = msgpack.packb(tagged_union[1:])
        
        if tag == BLOCK_STORED_TAG:
            event = msgpack.unpackb(payload_bytes, raw=False, strict_map_key=False)
            return BlockStored.from_msgpack(event), ""
                
        if tag == BLOCK_REMOVED_TAG:
            event = msgpack.unpackb(payload_bytes, raw=False, strict_map_key=False)
            return BlockRemoved.from_msgpack(event), ""
            
        if tag == ALL_BLOCKS_CLEARED_TAG:
            return AllBlocksCleared.from_msgpack([]), ""
            
        return None, f"unknown event tag: {tag}"
                
    except Exception as e:
        return None, f"unmarshal error: {str(e)}"


def parse_topic(topic: str) -> tuple[str, str]:
    """Extract pod_identifier and model_name from topic (kv@@pod@model format)"""
    parts = topic.split("@")
    if len(parts) == EXPECTED_MESSAGE_PARTS:
        return parts[1], parts[2]
    raise ValueError(f"Invalid topic format, expected kv@@, got: {topic}")

def print_msg_info(parts: Any):
    topic = parts[0].decode('utf-8')
    seq_bytes = parts[1]
    seq = struct.unpack(">Q", seq_bytes)[0]

    try:
        pod_id, model_name = parse_topic(topic)
        logger.info(f"Received ZMQ message, topic: {topic}, seq: {seq}, pod_id: {pod_id}, model_name: {model_name}")
    except ValueError as e:
        logger.error(f"Failed to parse topic {topic}: {e}")

def create_zmq_socket(context: zmq.Context, bind_address: str, max_retries: int) -> zmq.Socket|None:
    """Create and bind ZMQ socket with retry logic"""
    for attempt in range(max_retries):
        try:
            socket = context.socket(zmq.SUB)
            socket.bind(bind_address)
            socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all topics
            logger.info(f"Successfully bound to {bind_address}")
            return socket
        except zmq.ZMQError as e:
            logger.error(f"Failed to bind socket (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)  # Exponential backoff
            else:
                return None

def listener():
    port = os.getenv('ZMQ_PORT', '')
    if not port:
        logger.info("ZMQ_PORT environment variable is required")
        return
        
    bind_address = f"tcp://*:{port}"
    logger.info(f"Starting ZMQ listener on {bind_address}")
    
    context = zmq.Context()
    socket = create_zmq_socket(context, bind_address, MAX_BIND_RETRIES)

    if not socket:
        logger.info("Failed to bind socket - exiting")
        return
    
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        socket.close()
        context.term()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            socks = dict(poller.poll(POLL_TIMEOUT_MS))

            if socks.get(socket) == zmq.POLLIN:
                parts = socket.recv_multipart()
                if len(parts) != 3:
                    logger.error(f"Invalid message parts: {len(parts)}")
                    continue
                print_msg_info(parts)

                try:
                    payload = parts[2]
                    event_batch_raw = msgpack.unpackb(payload, raw=False, strict_map_key=False)                    
                    event_batch = EventBatch(
                        ts=event_batch_raw[0],
                        events=[msgpack.packb(e) for e in event_batch_raw[1]],
                        data_parallel_rank=event_batch_raw[2] if len(event_batch_raw) > 2 else None
                    )
                except Exception as e:
                    logger.error(f"Failed to unmarshal EventBatch: {e}")
                    continue

                # Process each event in batch
                for raw_event in event_batch.events:
                    event, err = unmarshal_kv_event(raw_event)
                    if err:
                        logger.error(f"Failed to unmarshal event: {err}")
                        continue
                    if not event:
                        logger.error(f"Failed to unmarshal event")
                        continue
                    
                    # Log event
                    logger.info(f"Event: {', '.join(f"{k}={v}" for k, v in event.to_dict().items())}")

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    listener()