# Streaming Features Documentation
## Optional Real-Time Response Streaming

The Modern RAG App now supports **optional streaming responses** for enhanced user experience and real-time interaction. This feature provides progressive response delivery while maintaining all the intelligent analysis and retrieval capabilities.

## 🎯 Overview

### What is Streaming?
- **Progressive delivery**: Response chunks sent as they're generated
- **Real-time interaction**: Users see responses building in real-time
- **Improved UX**: Better perceived performance for long responses
- **Optional feature**: Can be enabled/disabled per request or globally

### When to Use Streaming
- **Long responses**: Complex questions requiring detailed answers
- **Interactive applications**: Chat interfaces, live demos
- **User engagement**: Better experience for waiting users
- **Real-time systems**: Applications requiring immediate feedback

## 🚀 Implementation Details

### Architecture Overview
```
Query → Analysis → Retrieval → Streaming Summarization → Progressive Response
   ↓        ↓          ↓              ↓                       ↓
Status   Intent    Chunks      LLaMA/Templates        Real-time Chunks
```

### Streaming Pipeline
1. **Initialization**: Send metadata about the query and processing mode
2. **Status Updates**: Real-time updates during analysis and retrieval
3. **Progressive Response**: Content delivered in configurable chunks
4. **Completion**: Final metadata with sources and quality metrics

## 📋 Configuration Options

### Environment Variables
```bash
# Enable streaming globally
ENABLE_STREAMING=true

# Default streaming settings
STREAM_CHUNK_SIZE=50        # Words per chunk
STREAM_DELAY_MS=50          # Milliseconds between chunks
```

### Request-Level Configuration
```json
{
    "query": "What is the budget allocation?",
    "mode": "balanced",
    "stream_chunk_size": 70,    # Override default
    "stream_delay_ms": 30       # Faster streaming
}
```

### Processing Modes & Streaming
- **Speed Mode**: 30-word chunks, 30ms delay, template responses
- **Balanced Mode**: 50-word chunks, 50ms delay, LLaMA or enhanced templates  
- **Comprehensive Mode**: 70-word chunks, 70ms delay, full LLaMA processing

## 🔧 API Usage

### Standard (Non-Streaming) Endpoint
```bash
POST /ask
Content-Type: application/json

{
    "query": "What is the budget allocation?",
    "collection_name": "default",
    "mode": "balanced"
}

# Returns complete response immediately
```

### Streaming Endpoint
```bash
POST /ask/stream
Content-Type: application/json

{
    "query": "What is the budget allocation?",
    "collection_name": "default",
    "mode": "balanced",
    "stream_chunk_size": 50,
    "stream_delay_ms": 50
}

# Returns Server-Sent Events stream
```

### Server-Sent Events Format
```javascript
// Status update
data: {"type": "chunk", "chunk_id": 1, "content": "", "metadata": {"stage": "analysis"}}

// Content chunk
data: {"type": "chunk", "chunk_id": 4, "content": "Based on the budget documents", "metadata": {"progress": 0.1}}

// More content
data: {"type": "chunk", "chunk_id": 5, "content": ", the allocation shows three", "metadata": {"progress": 0.2}}

// Final response
data: {"type": "final", "response": {...}, "completed": true}
```

## 💻 Frontend Integration

### JavaScript/TypeScript Example
```javascript
async function streamingQuery(query) {
    const response = await fetch('/ask/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: query,
            mode: 'balanced',
            stream_chunk_size: 50,
            stream_delay_ms: 50
        })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let responseText = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'chunk') {
                    if (data.metadata?.stage) {
                        updateStatus(data.metadata.message);
                    } else {
                        responseText += data.content;
                        updateResponse(responseText);
                    }
                } else if (data.type === 'final') {
                    onComplete(data.response);
                }
            }
        }
    }
}

function updateStatus(message) {
    document.getElementById('status').textContent = message;
}

function updateResponse(text) {
    document.getElementById('response').textContent = text;
}

function onComplete(finalResponse) {
    document.getElementById('status').textContent = 'Complete';
    console.log('Sources:', finalResponse.sources_used);
    console.log('Confidence:', finalResponse.confidence_score);
}
```

### React Hook Example
```jsx
import { useState, useCallback } from 'react';

export const useStreamingQuery = () => {
    const [response, setResponse] = useState('');
    const [status, setStatus] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);
    const [finalData, setFinalData] = useState(null);

    const streamQuery = useCallback(async (query, options = {}) => {
        setIsStreaming(true);
        setResponse('');
        setStatus('Starting...');

        try {
            const res = await fetch('/ask/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    mode: 'balanced',
                    ...options
                })
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));

                        if (data.type === 'chunk') {
                            if (data.metadata?.stage) {
                                setStatus(data.metadata.message);
                            } else {
                                setResponse(prev => prev + data.content);
                            }
                        } else if (data.type === 'final') {
                            setFinalData(data.response);
                            setStatus('Complete');
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Streaming error:', error);
            setStatus('Error occurred');
        } finally {
            setIsStreaming(false);
        }
    }, []);

    return { response, status, isStreaming, finalData, streamQuery };
};
```

## ⚡ Performance Characteristics

### Streaming vs Non-Streaming Comparison
| Aspect | Non-Streaming | Streaming |
|--------|---------------|-----------|
| **First Response** | 2-5 seconds | 200-500ms |
| **User Perception** | Waiting | Progressive |
| **Memory Usage** | Peak at end | Distributed |
| **Error Recovery** | All-or-nothing | Graceful degradation |
| **Network** | Single large payload | Multiple small chunks |

### Optimal Settings by Use Case
| Use Case | Chunk Size | Delay | Mode | Best For |
|----------|------------|-------|------|----------|
| **Chat Interface** | 30 words | 30ms | Speed | Real-time feel |
| **Document Analysis** | 70 words | 50ms | Balanced | Quality + Speed |
| **Research Assistant** | 100 words | 70ms | Comprehensive | Maximum quality |
| **Mobile App** | 40 words | 60ms | Balanced | Network efficiency |

## 🛠️ Advanced Features

### Chunk Metadata
Each streaming chunk includes rich metadata:
```json
{
    "chunk_id": 5,
    "content": "The budget allocation shows",
    "metadata": {
        "type": "content",
        "progress": 0.15,
        "confidence": 0.85,
        "sources_preview": ["budget_2024.pdf"],
        "reasoning_step": "Analyzing financial data"
    }
}
```

### Error Handling
Streaming includes robust error handling:
```json
{
    "type": "error",
    "error": "Processing timeout",
    "partial_response": "Based on the available data...",
    "recovery_options": ["retry", "fallback_mode"]
}
```

### Quality Metrics
Real-time quality indicators during streaming:
- **Progress tracking**: Percentage completion
- **Confidence scoring**: Per-chunk confidence
- **Source attribution**: Live source tracking
- **Coherence monitoring**: Response quality metrics

## 🔒 Production Considerations

### Security
- **Rate limiting**: Prevent streaming abuse
- **Authentication**: Secure streaming endpoints
- **Input validation**: Sanitize streaming parameters
- **Resource monitoring**: Track concurrent streams

### Scalability
- **Connection pooling**: Manage SSE connections
- **Load balancing**: Distribute streaming load
- **Caching**: Cache partial responses when appropriate  
- **Resource limits**: Set maximum concurrent streams

### Monitoring
- **Stream metrics**: Track chunk delivery times
- **Error rates**: Monitor streaming failures
- **User engagement**: Measure streaming effectiveness
- **Performance impact**: Compare streaming vs non-streaming costs

## 🔧 Troubleshooting

### Common Issues
1. **Slow streaming**: Reduce chunk size, decrease delay
2. **Memory issues**: Enable garbage collection, reduce batch sizes
3. **Network timeouts**: Implement connection retry logic
4. **Frontend buffering**: Ensure proper SSE handling

### Debug Configuration
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export STREAM_DEBUG=true

# Test with curl
curl -N -X POST "http://localhost:8000/ask/stream" \
     -H "Content-Type: application/json" \
     -d '{"query": "test", "mode": "speed"}'
```

## 📈 Future Enhancements

### Planned Features
- **Bi-directional streaming**: Interactive conversations
- **Stream branching**: Multiple response alternatives
- **Quality adaptation**: Dynamic quality based on network
- **Streaming analytics**: Advanced metrics and insights

### Integration Opportunities
- **WebSocket support**: Lower latency streaming
- **GraphQL subscriptions**: Structured streaming queries
- **gRPC streaming**: High-performance binary streaming
- **Real-time collaboration**: Multi-user streaming sessions

## 🎯 Best Practices

### When to Enable Streaming
✅ **Good for:**
- Interactive applications
- Long-form responses
- Real-time user feedback
- Mobile applications
- Live demos

❌ **Avoid for:**
- Simple yes/no questions
- Batch processing
- API integrations without UI
- Systems with strict latency requirements

### Configuration Tips
1. **Start conservative**: Begin with balanced mode, standard settings
2. **Monitor performance**: Track both streaming and system metrics
3. **User testing**: Validate streaming improves actual user experience
4. **Fallback planning**: Always support non-streaming as fallback
5. **Error handling**: Implement comprehensive error recovery

The streaming feature enhances user experience while maintaining the full power and accuracy of the Modern RAG system. It's designed to be completely optional and can be enabled per-request or globally based on your application needs.
