# Intelligent Summarizer Service

The Intelligent Summarizer service provides advanced text summarization and response generation capabilities using LLaMA models, with context-aware processing based on query intent and retrieved content.

## Overview

The Intelligent Summarizer (`services/intelligent_summarizer.py`) is responsible for generating human-like responses by combining retrieved document chunks with query context through Large Language Model (LLM) processing. It supports multiple response styles and processing modes optimized for different use cases.

## Key Responsibilities

### 🧠 **Intelligent Response Generation**
- Context-aware summarization based on query intent
- Integration of multiple retrieved document chunks
- Response style adaptation (factual, analytical, comparative, etc.)
- Source attribution and citation management

### 🎯 **Query-Adaptive Processing**
- Response tailoring based on query analysis results
- Context expansion for comprehensive answers
- Relevance-based content prioritization
- Multi-document synthesis capabilities

### ⚡ **Performance Optimization**
- Multiple processing modes (Fast/Balanced/Comprehensive)
- Streaming response capabilities for real-time interaction
- Response caching and optimization
- Resource-efficient LLM utilization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Intelligent Summarizer                       │
├─────────────────────────────────────────────────────────────┤
│  Input Processing                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Query     │  │ Retrieved   │  │   Context   │        │
│  │  Analysis   │  │   Chunks    │  │ Enhancement │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  LLM Processing                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Prompt    │  │    LLaMA    │  │  Response   │        │
│  │ Generation  │  │  Processing │  │ Processing  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Processing Modes

### ⚡ **Fast Mode** (200-500ms)
- **Template-Based**: Pre-defined response templates
- **Basic Context**: Limited context integration
- **Simple Citations**: Basic source attribution
- **Use Case**: Quick responses, real-time chat

### ⚖️ **Balanced Mode** (500-1500ms)  
- **Dynamic Prompts**: Query-adaptive prompt generation
- **Full Context**: Complete retrieved content integration
- **Smart Citations**: Intelligent source selection and attribution
- **Use Case**: Standard queries, balanced performance

### 🔍 **Comprehensive Mode** (1500-5000ms)
- **Advanced Analysis**: Deep content analysis and synthesis
- **Multi-Document**: Cross-document relationship analysis
- **Rich Citations**: Detailed source attribution with metadata
- **Use Case**: Complex analysis, research queries

## Response Styles

### 📊 **Factual Style**
Direct, data-focused responses with clear citations:
```
"According to the financial report (Q3-2024.pdf, page 12), the company's revenue increased by 15% to $2.3 million in Q3 2024."
```

### 🔍 **Analytical Style**  
Detailed explanations with reasoning and context:
```
"The revenue increase can be attributed to three main factors: 1) Expanded market presence in Asia (Marketing-Strategy.pdf), 2) New product line introduction (Product-Launch.pdf), and 3) Improved operational efficiency (Operations-Report.pdf). This growth pattern suggests..."
```

### ⚖️ **Comparative Style**
Side-by-side analysis of different options or viewpoints:
```
"Comparing the two approaches: Method A shows 23% efficiency gains but requires $50K investment, while Method B offers 18% gains with only $20K investment. The ROI analysis indicates..."
```

### 📝 **Procedural Style**
Step-by-step instructions and process descriptions:
```
"To implement the new system: 1. Install dependencies (Setup-Guide.pdf, Section 2), 2. Configure environment variables (Config-Manual.pdf), 3. Run initialization scripts..."
```

## Key Methods

### `async def generate_response()`
Main response generation with full LLM processing:

```python
async def generate_response(
    self,
    query_analysis: QueryAnalysis,
    retrieval_result: RetrievalResult,
    config: SummarizationConfig
) -> SummarizationResult
```

**Features:**
- Context-aware response generation
- Multiple response style support
- Comprehensive source attribution
- Performance mode optimization

### `async def stream_response()`
Streaming response generation for real-time interaction:

```python
async def stream_response(
    self,
    query_analysis: QueryAnalysis,
    retrieval_result: RetrievalResult,
    config: SummarizationConfig
) -> AsyncGenerator[StreamChunk, None]
```

**Features:**
- Real-time token streaming
- Progressive response building
- Cancellation support
- Error handling during streaming

### `async def synthesize_multi_document()`
Advanced multi-document synthesis and analysis:

```python
async def synthesize_multi_document(
    self,
    documents: List[RetrievalResult],
    synthesis_type: SynthesisType,
    config: SummarizationConfig
) -> SynthesisResult
```

**Features:**
- Cross-document relationship analysis
- Comprehensive information synthesis
- Conflict resolution and fact checking
- Advanced citation management

## Configuration

### SummarizationConfig
```python
@dataclass
class SummarizationConfig:
    mode: SummarizationMode = SummarizationMode.BALANCED
    style: ResponseStyle = ResponseStyle.FACTUAL
    max_tokens: int = 500
    include_citations: bool = True
    streaming: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
```

### Environment Configuration
```bash
# LLM Configuration
LLAMA_MODEL_PATH=/path/to/llama/model
LLAMA_DEVICE=auto  # auto, cpu, cuda
LLAMA_MAX_TOKENS=2048

# Performance Settings
SUMMARIZER_DEFAULT_MODE=balanced
SUMMARIZER_MAX_CONCURRENT=10
SUMMARIZER_TIMEOUT_SECONDS=30

# Response Configuration
SUMMARIZER_DEFAULT_STYLE=factual
SUMMARIZER_INCLUDE_CITATIONS=true
```

## Prompt Engineering

### Context Integration Strategy
```python
def build_context_prompt(
    query: str,
    chunks: List[RankedChunk],
    style: ResponseStyle
) -> str:
    """
    Builds optimized prompts based on:
    - Query intent and complexity
    - Retrieved content relevance
    - Desired response style
    - Available context length
    """
```

### Citation Management
```python
def generate_citations(
    response: str,
    source_chunks: List[RankedChunk]
) -> List[Citation]:
    """
    Generates accurate citations by:
    - Mapping response content to source chunks
    - Creating proper attribution formats
    - Handling conflicting information
    - Maintaining citation consistency
    """
```

## Streaming Features

### Real-Time Response Generation
```python
async def stream_example():
    async for chunk in summarizer.stream_response(
        query_analysis=analysis,
        retrieval_result=results,
        config=streaming_config
    ):
        if chunk.type == "token":
            print(chunk.content, end="", flush=True)
        elif chunk.type == "citation":
            print(f"\n[Source: {chunk.source}]")
        elif chunk.type == "complete":
            print(f"\n[Response complete: {chunk.metadata}]")
```

### Stream Chunk Types
- **Token**: Individual response tokens for real-time display
- **Citation**: Source attribution information
- **Metadata**: Processing status and statistics
- **Error**: Error information with recovery suggestions
- **Complete**: Final response metadata and statistics

## Integration Points

### 🔗 **Service Dependencies**
- **Query Analyzer**: Provides intent classification for style selection
- **Intelligent Retriever**: Supplies ranked chunks for content synthesis
- **LLM Services**: Ollama, OpenAI, or custom LLM providers
- **Vector Store**: Access to document metadata and relationships

### 📊 **Response Quality Management**
- Response coherence validation
- Citation accuracy checking
- Content relevance scoring
- Hallucination detection and prevention

## Performance Characteristics

| Mode | Response Time | Quality Score | Token Throughput | Memory Usage |
|------|---------------|---------------|------------------|--------------|
| Fast | 200-500ms | 7/10 | 100+ tokens/s | Low |
| Balanced | 500-1500ms | 8.5/10 | 50-80 tokens/s | Moderate |
| Comprehensive | 1500-5000ms | 9.5/10 | 20-40 tokens/s | High |

## Error Handling

### 🛡️ **Graceful Degradation**
- LLM failure fallback to template-based responses
- Partial content generation when context is limited
- Citation recovery from source metadata
- Performance mode downgrade under resource constraints

### 🔄 **Recovery Strategies**
- Automatic retry with different parameters
- Alternative prompt strategies
- Fallback to simpler response styles
- Content validation and correction

## Usage Examples

### Basic Response Generation
```python
summarizer = IntelligentSummarizer()

# Generate response
result = await summarizer.generate_response(
    query_analysis=query_analysis,
    retrieval_result=retrieval_results,
    config=SummarizationConfig(
        mode=SummarizationMode.BALANCED,
        style=ResponseStyle.ANALYTICAL,
        max_tokens=300
    )
)

print(result.response)
print(f"Citations: {len(result.citations)}")
```

### Streaming Response
```python
config = SummarizationConfig(streaming=True)

async for chunk in summarizer.stream_response(
    query_analysis=analysis,
    retrieval_result=results,
    config=config
):
    handle_stream_chunk(chunk)
```

### Multi-Document Synthesis
```python
# Synthesize across multiple documents
synthesis_result = await summarizer.synthesize_multi_document(
    documents=multiple_retrieval_results,
    synthesis_type=SynthesisType.COMPREHENSIVE_ANALYSIS,
    config=comprehensive_config
)

print(synthesis_result.synthesized_response)
print(f"Documents analyzed: {len(synthesis_result.source_documents)}")
```

## Quality Assurance

### Response Validation
- **Factual Consistency**: Verify claims against source documents
- **Citation Accuracy**: Ensure proper attribution and references
- **Coherence Check**: Validate logical flow and readability
- **Completeness Score**: Measure response comprehensiveness

### Continuous Improvement
- Response quality feedback collection
- Performance metric monitoring
- Model fine-tuning based on usage patterns
- A/B testing for prompt optimization

## Related Documentation
- [RAG Orchestrator](rag_orchestrator.md) - Complete pipeline coordination
- [Intelligent Retriever](intelligent_retriever.md) - Content retrieval and ranking
- [Query Analyzer](query_analyzer.md) - Intent classification for style selection
- [Configuration Guide](../configuration.md) - LLM and performance settings

---

**The Intelligent Summarizer transforms retrieved content into coherent, contextual responses that directly address user queries with appropriate style and comprehensive source attribution.**
