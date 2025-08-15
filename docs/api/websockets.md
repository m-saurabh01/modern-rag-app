# WebSocket Communication Guide

## 📋 Overview

The Modern RAG Application provides real-time WebSocket connections for interactive query processing, document upload progress, and system monitoring. This enables responsive user interfaces and streaming data flows.

## 🔌 Connection Endpoints

### **Primary WebSocket Endpoints**
- `ws://localhost:8000/ws/query` - Real-time query processing
- `ws://localhost:8000/ws/upload` - Document upload progress  
- `ws://localhost:8000/ws/monitor` - System monitoring and metrics
- `ws://localhost:8000/ws/collections` - Collection updates and notifications

## 🚀 Query Processing WebSocket

### **Connection**
```javascript
// Basic connection
const ws = new WebSocket('ws://localhost:8000/ws/query');

// With authentication
const ws = new WebSocket('ws://localhost:8000/ws/query', [], {
    headers: {
        'Authorization': 'Bearer your-jwt-token'
    }
});
```

### **Message Protocol**

#### **Client → Server Messages**

**Query Message:**
```json
{
    "type": "query",
    "request_id": "req_12345",
    "data": {
        "query": "What is the budget for IT department?",
        "collection_name": "documents",
        "mode": "balanced",
        "stream": true,
        "max_results": 10,
        "include_sources": true
    }
}
```

**Configuration Message:**
```json
{
    "type": "config",
    "request_id": "req_12346", 
    "data": {
        "response_format": "with_citations",
        "language": "en",
        "temperature": 0.7
    }
}
```

**Control Message:**
```json
{
    "type": "control",
    "request_id": "req_12347",
    "data": {
        "action": "cancel"  // "pause", "resume", "cancel"
    }
}
```

#### **Server → Client Messages**

**Status Updates:**
```json
{
    "type": "status",
    "request_id": "req_12345",
    "data": {
        "stage": "analyzing_query",
        "progress": 0.1,
        "message": "Analyzing query intent..."
    }
}
```

**Streaming Response Tokens:**
```json
{
    "type": "token",
    "request_id": "req_12345",
    "data": {
        "content": "The",
        "position": 0,
        "is_complete": false
    }
}
```

**Citations and Sources:**
```json
{
    "type": "citation",
    "request_id": "req_12345",
    "data": {
        "citation_id": 1,
        "source": "budget_report_2024.pdf",
        "page": 15,
        "confidence": 0.92,
        "content": "IT department budget allocation..."
    }
}
```

**Completion Message:**
```json
{
    "type": "complete",
    "request_id": "req_12345",
    "data": {
        "total_time": 2.3,
        "tokens_generated": 256,
        "sources_used": 3,
        "processing_mode": "balanced",
        "query_analysis": {
            "intent": "factual",
            "entities": ["IT department", "budget"]
        }
    }
}
```

**Error Message:**
```json
{
    "type": "error",
    "request_id": "req_12345",
    "data": {
        "code": "PROCESSING_ERROR",
        "message": "Failed to process query",
        "details": {
            "stage": "document_retrieval",
            "error": "Connection timeout"
        },
        "recoverable": true
    }
}
```

### **JavaScript Client Example**

```javascript
class RAGWebSocketClient {
    constructor(url, authToken = null) {
        this.url = url;
        this.authToken = authToken;
        this.ws = null;
        this.requestCallbacks = new Map();
    }
    
    connect() {
        const protocols = [];
        const headers = this.authToken ? 
            { 'Authorization': `Bearer ${this.authToken}` } : {};
        
        this.ws = new WebSocket(this.url, protocols);
        
        this.ws.onopen = () => {
            console.log('Connected to RAG WebSocket');
            this.onConnected?.();
        };
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };
        
        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            this.onDisconnected?.(event);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.onError?.(error);
        };
    }
    
    handleMessage(message) {
        const { type, request_id, data } = message;
        
        const callback = this.requestCallbacks.get(request_id);
        if (callback) {
            callback(type, data);
        }
        
        // Global message handlers
        switch (type) {
            case 'status':
                this.onStatus?.(request_id, data);
                break;
            case 'token':
                this.onToken?.(request_id, data);
                break;
            case 'citation':
                this.onCitation?.(request_id, data);
                break;
            case 'complete':
                this.onComplete?.(request_id, data);
                this.requestCallbacks.delete(request_id);
                break;
            case 'error':
                this.onError?.(request_id, data);
                this.requestCallbacks.delete(request_id);
                break;
        }
    }
    
    query(queryText, options = {}, callback = null) {
        const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        if (callback) {
            this.requestCallbacks.set(requestId, callback);
        }
        
        const message = {
            type: 'query',
            request_id: requestId,
            data: {
                query: queryText,
                stream: true,
                ...options
            }
        };
        
        this.send(message);
        return requestId;
    }
    
    cancelQuery(requestId) {
        this.send({
            type: 'control',
            request_id: requestId,
            data: { action: 'cancel' }
        });
    }
    
    send(message) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.error('WebSocket not connected');
        }
    }
}

// Usage example
const client = new RAGWebSocketClient('ws://localhost:8000/ws/query', 'your-jwt-token');

client.onConnected = () => console.log('Ready for queries');
client.onStatus = (requestId, status) => updateProgressBar(status.progress);
client.onToken = (requestId, token) => appendToResponse(token.content);
client.onCitation = (requestId, citation) => addCitation(citation);
client.onComplete = (requestId, completion) => finalizeResponse(completion);

client.connect();

// Send query
const queryId = client.query('What is the IT budget?', {
    mode: 'comprehensive',
    max_results: 5
});
```

### **Python Client Example**

```python
import asyncio
import websockets
import json
from typing import Optional, Callable

class RAGWebSocketClient:
    def __init__(self, uri: str, auth_token: Optional[str] = None):
        self.uri = uri
        self.auth_token = auth_token
        self.websocket = None
        self.request_callbacks = {}
    
    async def connect(self):
        headers = {}
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        
        self.websocket = await websockets.connect(
            self.uri,
            extra_headers=headers
        )
        
        # Start message handler
        asyncio.create_task(self._handle_messages())
    
    async def _handle_messages(self):
        async for message in self.websocket:
            data = json.loads(message)
            await self._process_message(data)
    
    async def _process_message(self, message):
        msg_type = message['type']
        request_id = message['request_id']
        data = message['data']
        
        # Call specific callback if registered
        if request_id in self.request_callbacks:
            callback = self.request_callbacks[request_id]
            await callback(msg_type, data)
        
        # Handle completion cleanup
        if msg_type in ['complete', 'error']:
            self.request_callbacks.pop(request_id, None)
    
    async def query(
        self, 
        query_text: str, 
        options: dict = None,
        callback: Callable = None
    ) -> str:
        request_id = f"req_{asyncio.get_event_loop().time()}_{id(self)}"
        
        if callback:
            self.request_callbacks[request_id] = callback
        
        message = {
            "type": "query",
            "request_id": request_id,
            "data": {
                "query": query_text,
                "stream": True,
                **(options or {})
            }
        }
        
        await self.websocket.send(json.dumps(message))
        return request_id
    
    async def cancel_query(self, request_id: str):
        message = {
            "type": "control",
            "request_id": request_id,
            "data": {"action": "cancel"}
        }
        await self.websocket.send(json.dumps(message))

# Usage example
async def query_handler(msg_type, data):
    if msg_type == 'status':
        print(f"Status: {data['message']}")
    elif msg_type == 'token':
        print(data['content'], end='', flush=True)
    elif msg_type == 'citation':
        print(f"\n[{data['citation_id']}] {data['source']}")
    elif msg_type == 'complete':
        print(f"\n\nCompleted in {data['total_time']:.2f}s")

async def main():
    client = RAGWebSocketClient('ws://localhost:8000/ws/query')
    await client.connect()
    
    query_id = await client.query(
        "What is the budget allocation?",
        options={"mode": "comprehensive"},
        callback=query_handler
    )
    
    # Keep connection alive
    await asyncio.sleep(10)

asyncio.run(main())
```

## 📤 Upload Progress WebSocket

### **Connection**
```javascript
const uploadWs = new WebSocket('ws://localhost:8000/ws/upload');
```

### **Upload Progress Messages**

```json
// File upload started
{
    "type": "upload_started",
    "data": {
        "upload_id": "upload_12345",
        "filename": "document.pdf",
        "file_size": 2048576,
        "estimated_time": 15
    }
}

// Upload progress
{
    "type": "upload_progress",
    "data": {
        "upload_id": "upload_12345",
        "filename": "document.pdf", 
        "bytes_uploaded": 1024000,
        "total_bytes": 2048576,
        "progress": 0.5,
        "speed_mbps": 1.2
    }
}

// Processing stages
{
    "type": "processing_progress",
    "data": {
        "upload_id": "upload_12345",
        "filename": "document.pdf",
        "stage": "text_extraction",
        "progress": 0.3,
        "message": "Extracting text from PDF..."
    }
}

// Chunking progress
{
    "type": "chunking_progress", 
    "data": {
        "upload_id": "upload_12345",
        "filename": "document.pdf",
        "chunks_processed": 8,
        "total_chunks": 15,
        "progress": 0.53
    }
}

// Embedding progress
{
    "type": "embedding_progress",
    "data": {
        "upload_id": "upload_12345", 
        "filename": "document.pdf",
        "embeddings_generated": 10,
        "total_embeddings": 15,
        "progress": 0.67
    }
}

// Upload complete
{
    "type": "upload_complete",
    "data": {
        "upload_id": "upload_12345",
        "filename": "document.pdf",
        "document_id": "doc_456",
        "total_time": 12.5,
        "chunks_created": 15,
        "status": "success"
    }
}
```

### **Upload Progress Handler**

```javascript
class UploadProgressTracker {
    constructor() {
        this.ws = new WebSocket('ws://localhost:8000/ws/upload');
        this.activeUploads = new Map();
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleUploadMessage(message);
        };
    }
    
    handleUploadMessage(message) {
        const { type, data } = message;
        const uploadId = data.upload_id;
        
        switch (type) {
            case 'upload_started':
                this.activeUploads.set(uploadId, {
                    filename: data.filename,
                    fileSize: data.file_size,
                    startTime: Date.now()
                });
                this.onUploadStarted?.(uploadId, data);
                break;
                
            case 'upload_progress':
                this.updateProgressBar(uploadId, data.progress, 'Uploading');
                this.onUploadProgress?.(uploadId, data);
                break;
                
            case 'processing_progress':
                this.updateProgressBar(uploadId, data.progress, data.message);
                this.onProcessingProgress?.(uploadId, data);
                break;
                
            case 'chunking_progress':
                this.updateProgressBar(uploadId, data.progress, 'Creating chunks');
                this.onChunkingProgress?.(uploadId, data);
                break;
                
            case 'embedding_progress':
                this.updateProgressBar(uploadId, data.progress, 'Generating embeddings');
                this.onEmbeddingProgress?.(uploadId, data);
                break;
                
            case 'upload_complete':
                this.completeUpload(uploadId, data);
                this.onUploadComplete?.(uploadId, data);
                break;
        }
    }
    
    updateProgressBar(uploadId, progress, message) {
        const progressBar = document.querySelector(`#progress-${uploadId}`);
        if (progressBar) {
            progressBar.style.width = `${progress * 100}%`;
            progressBar.textContent = `${message} (${Math.round(progress * 100)}%)`;
        }
    }
}
```

## 📊 System Monitoring WebSocket

### **Connection**
```javascript
const monitorWs = new WebSocket('ws://localhost:8000/ws/monitor');
```

### **Monitoring Messages**

```json
// System metrics
{
    "type": "system_metrics",
    "timestamp": "2024-08-15T10:30:00Z",
    "data": {
        "cpu_usage": 45.2,
        "memory_usage": 68.5,
        "active_connections": 12,
        "queries_per_minute": 25,
        "average_response_time": 1.2
    }
}

// Service status updates
{
    "type": "service_status",
    "timestamp": "2024-08-15T10:30:00Z",
    "data": {
        "service": "vector_database",
        "status": "healthy",
        "response_time": 15,
        "last_check": "2024-08-15T10:29:45Z"
    }
}

// Error alerts
{
    "type": "error_alert",
    "timestamp": "2024-08-15T10:30:00Z", 
    "data": {
        "severity": "warning",
        "service": "embedding_service",
        "message": "High memory usage detected",
        "details": {
            "memory_usage": 85.2,
            "threshold": 80.0
        }
    }
}
```

## 🔄 Collection Updates WebSocket

### **Real-time Collection Changes**

```json
// Document added
{
    "type": "document_added",
    "collection": "research_papers",
    "data": {
        "document_id": "doc_789",
        "filename": "new_research.pdf",
        "chunks_added": 18,
        "timestamp": "2024-08-15T10:30:00Z"
    }
}

// Collection statistics updated
{
    "type": "collection_stats",
    "collection": "research_papers", 
    "data": {
        "total_documents": 51,
        "total_chunks": 768,
        "storage_size_mb": 127.3,
        "last_updated": "2024-08-15T10:30:00Z"
    }
}
```

## 🚨 Error Handling

### **Connection Errors**
```javascript
ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    
    // Attempt reconnection
    setTimeout(() => {
        reconnectWebSocket();
    }, 5000);
};

ws.onclose = (event) => {
    if (event.code !== 1000) {  // Not a normal closure
        console.log('Connection lost, attempting to reconnect...');
        reconnectWebSocket();
    }
};

function reconnectWebSocket() {
    // Exponential backoff
    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
    
    setTimeout(() => {
        connectWebSocket();
        reconnectAttempts++;
    }, delay);
}
```

### **Message Validation**
```javascript
function validateMessage(message) {
    try {
        const parsed = JSON.parse(message);
        
        if (!parsed.type || !parsed.request_id) {
            throw new Error('Invalid message format');
        }
        
        return parsed;
    } catch (error) {
        console.error('Message validation failed:', error);
        return null;
    }
}
```

## 🔐 Authentication

### **JWT Authentication**
```javascript
// Include JWT token in connection
const token = localStorage.getItem('auth_token');
const ws = new WebSocket('ws://localhost:8000/ws/query', [], {
    headers: {
        'Authorization': `Bearer ${token}`
    }
});
```

### **Authentication Failure**
```json
{
    "type": "auth_error",
    "data": {
        "code": "INVALID_TOKEN",
        "message": "Authentication failed",
        "action": "reconnect_with_valid_token"
    }
}
```

## 🧪 Testing WebSocket Connections

### **Unit Tests**
```python
import pytest
import asyncio
import websockets
import json

@pytest.mark.asyncio
async def test_query_websocket():
    uri = "ws://localhost:8000/ws/query"
    
    async with websockets.connect(uri) as websocket:
        # Send query
        query_message = {
            "type": "query",
            "request_id": "test_123",
            "data": {"query": "test query", "stream": True}
        }
        
        await websocket.send(json.dumps(query_message))
        
        # Receive and validate response
        response = await websocket.recv()
        data = json.loads(response)
        
        assert data['type'] in ['status', 'token', 'complete']
        assert data['request_id'] == 'test_123'
```

### **Load Testing**
```javascript
// Simulate multiple concurrent connections
async function loadTestWebSocket() {
    const connections = [];
    
    for (let i = 0; i < 50; i++) {
        const ws = new WebSocket('ws://localhost:8000/ws/query');
        connections.push(ws);
        
        ws.onopen = () => {
            ws.send(JSON.stringify({
                type: 'query',
                request_id: `load_test_${i}`,
                data: { query: `Load test query ${i}` }
            }));
        };
    }
    
    // Monitor performance
    setTimeout(() => {
        connections.forEach(ws => ws.close());
    }, 10000);
}
```

## 📚 Related Documentation

- **[API Endpoints](endpoints.md)** - REST API reference
- **[Authentication](authentication.md)** - Security and auth
- **[Rate Limiting](rate_limiting.md)** - Connection limits
- **[Error Handling](../core/exception_handling.md)** - Error management

---

**WebSocket connections provide real-time, interactive communication for responsive RAG applications.**
