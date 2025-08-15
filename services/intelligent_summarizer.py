"""
Intelligent Summarizer Service - Phase 3.4

This service provides intelligent summarization capabilities using LLaMA models
with context-aware summarization based on query intent and retrieved content.
Integrates with all previous phases for complete RAG pipeline.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import logging
import json

# Placeholder imports - would be actual integrations in production
try:
    import torch
    from transformers import LlamaForCausalLM, LlamaTokenizer, TextStreamer
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logging.warning("LLaMA dependencies not available. Using mock responses.")

from models.query_models import QueryAnalysis
from models.retrieval_models import RetrievalResult, RankedChunk
from services.intelligent_retriever import IntelligentRetriever
from services.query_analyzer import QueryAnalyzer


class SummarizationMode(Enum):
    """Summarization performance modes"""
    FAST = "fast"           # Quick summaries with basic templates
    BALANCED = "balanced"   # Good quality with moderate processing
    COMPREHENSIVE = "comprehensive"  # Maximum quality with full analysis


class ResponseStyle(Enum):
    """Response style based on query intent"""
    FACTUAL = "factual"           # Direct, data-focused answers
    ANALYTICAL = "analytical"     # Detailed explanations with reasoning
    COMPARATIVE = "comparative"   # Side-by-side comparisons
    PROCEDURAL = "procedural"     # Step-by-step instructions
    CONVERSATIONAL = "conversational"  # Natural dialogue style


@dataclass
class SummarizationConfig:
    """Configuration for intelligent summarization"""
    mode: SummarizationMode = SummarizationMode.BALANCED
    response_style: ResponseStyle = ResponseStyle.CONVERSATIONAL
    max_response_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Context management
    max_context_tokens: int = 2048
    context_compression_ratio: float = 0.7
    
    # Citation and source handling
    include_citations: bool = True
    citation_style: str = "inline"  # inline, footnote, bracketed
    
    # Quality controls
    enable_fact_checking: bool = True
    enable_coherence_check: bool = True
    min_confidence_score: float = 0.6
    
    # Streaming configuration
    enable_streaming: bool = False
    stream_chunk_size: int = 50  # Words per chunk
    stream_delay_ms: int = 50   # Delay between chunks for smooth streaming


@dataclass
class SummarizedResponse:
    """Complete summarized response with metadata"""
    response_text: str
    confidence_score: float
    sources_used: List[str]
    citations: List[Dict[str, Any]]
    
    # Processing metadata
    processing_time_ms: float
    tokens_generated: int
    context_chunks_used: int
    
    # Quality metrics
    coherence_score: float
    factual_consistency_score: float
    style_alignment_score: float
    
    # Reasoning chain (for transparency)
    reasoning_steps: List[str]
    context_summary: str
    
    # Streaming metadata
    is_streaming: bool = False
    stream_id: Optional[str] = None


@dataclass
class StreamChunk:
    """Individual streaming chunk"""
    chunk_id: int
    content: str
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class IntelligentSummarizer:
    """
    Intelligent Summarizer with LLaMA integration
    
    Provides context-aware summarization that adapts response style
    based on query intent and intelligently processes retrieved content.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "auto"):
        """Initialize the intelligent summarizer"""
        self.logger = logging.getLogger(__name__)
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.cache = {}  # Simple response cache
        
        # Load model if available
        if LLAMA_AVAILABLE and model_path:
            self._load_llama_model(model_path)
        else:
            self.logger.warning("Using mock LLaMA model for demonstration")
    
    def _setup_device(self, device: str) -> str:
        """Setup optimal device for model inference"""
        if device == "auto":
            if LLAMA_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            elif LLAMA_AVAILABLE and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_llama_model(self, model_path: str):
        """Load LLaMA model and tokenizer"""
        try:
            self.logger.info(f"Loading LLaMA model from {model_path}")
            self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
            self.model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device
            )
            self.logger.info("LLaMA model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load LLaMA model: {e}")
            self.model = None
            self.tokenizer = None
    
    async def summarize_with_context(self,
                                   query: str,
                                   query_analysis: QueryAnalysis,
                                   retrieval_result: RetrievalResult,
                                   config: Optional[SummarizationConfig] = None) -> Union[SummarizedResponse, AsyncGenerator[StreamChunk, None]]:
        """
        Generate intelligent summary based on query and retrieved context
        
        Args:
            query: Original user query
            query_analysis: Analyzed query with intent, entities, etc.
            retrieval_result: Results from intelligent retrieval
            config: Summarization configuration
            
        Returns:
            SummarizedResponse (complete) or AsyncGenerator[StreamChunk] (streaming)
        """
        start_time = time.time()
        config = config or SummarizationConfig()
        
        try:
            # 1. Adapt configuration based on query analysis
            adapted_config = self._adapt_config_to_query(config, query_analysis)
            
            # 2. Process and compress context
            processed_context = await self._process_context(
                retrieval_result.ranked_chunks, 
                adapted_config
            )
            
            # 3. Check for streaming mode
            if adapted_config.enable_streaming:
                return self._generate_streaming_response(
                    query, query_analysis, processed_context, adapted_config, retrieval_result
                )
            
            # 4. Generate complete response based on mode
            if adapted_config.mode == SummarizationMode.FAST:
                response = await self._generate_fast_summary(
                    query, processed_context, adapted_config
                )
            elif adapted_config.mode == SummarizationMode.COMPREHENSIVE:
                response = await self._generate_comprehensive_summary(
                    query, query_analysis, processed_context, adapted_config
                )
            else:  # BALANCED
                response = await self._generate_balanced_summary(
                    query, query_analysis, processed_context, adapted_config
                )
            
            # 5. Post-process and add citations
            final_response = await self._post_process_response(
                response, retrieval_result.ranked_chunks, adapted_config
            )
            
            # 6. Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(processing_time)
            
            return SummarizedResponse(
                response_text=final_response['text'],
                confidence_score=final_response['confidence'],
                sources_used=final_response['sources'],
                citations=final_response['citations'],
                processing_time_ms=processing_time,
                tokens_generated=final_response['token_count'],
                context_chunks_used=len(retrieval_result.ranked_chunks),
                coherence_score=final_response['coherence_score'],
                factual_consistency_score=final_response['factual_score'],
                style_alignment_score=final_response['style_score'],
                reasoning_steps=final_response['reasoning'],
                context_summary=processed_context['summary'],
                is_streaming=False
            )
            
        except Exception as e:
            self.logger.error(f"Summarization error: {e}")
            if config.enable_streaming:
                return self._create_error_stream(str(e))
            else:
                return self._create_error_response(str(e), time.time() - start_time)
    
    def _adapt_config_to_query(self, 
                              config: SummarizationConfig, 
                              query_analysis: QueryAnalysis) -> SummarizationConfig:
        """Adapt configuration based on query analysis"""
        adapted = config
        
        # Adapt response style based on intent
        if query_analysis.intent.name == "FACTUAL":
            adapted.response_style = ResponseStyle.FACTUAL
            adapted.temperature = 0.3  # More deterministic
        elif query_analysis.intent.name == "ANALYTICAL":
            adapted.response_style = ResponseStyle.ANALYTICAL
            adapted.max_response_length = 768  # Longer for explanations
        elif query_analysis.intent.name == "COMPARATIVE":
            adapted.response_style = ResponseStyle.COMPARATIVE
            adapted.max_response_length = 640  # Medium length for comparisons
        elif query_analysis.intent.name == "PROCEDURAL":
            adapted.response_style = ResponseStyle.PROCEDURAL
            adapted.temperature = 0.4  # Structured responses
        
        # Adapt length based on query complexity
        if len(query_analysis.entities) > 5:
            adapted.max_response_length = min(adapted.max_response_length * 1.3, 1024)
        
        return adapted
    
    async def _process_context(self, 
                             chunks: List[RankedChunk], 
                             config: SummarizationConfig) -> Dict[str, Any]:
        """Process and compress retrieved context for optimal summarization"""
        
        # Sort chunks by relevance
        sorted_chunks = sorted(chunks, key=lambda x: x.final_rank_score, reverse=True)
        
        # Extract and prioritize content
        prioritized_content = []
        tables_content = []
        source_metadata = []
        
        for chunk in sorted_chunks:
            content_info = {
                'text': chunk.chunk.content,
                'score': chunk.final_rank_score,
                'type': chunk.chunk.content_type,
                'source': chunk.chunk.metadata.get('source', 'Unknown'),
                'page': chunk.chunk.metadata.get('page_number', 0)
            }
            
            if chunk.chunk.content_type == 'table':
                tables_content.append(content_info)
            else:
                prioritized_content.append(content_info)
            
            source_metadata.append({
                'source': content_info['source'],
                'page': content_info['page'],
                'relevance': chunk.final_rank_score
            })
        
        # Compress context if needed
        compressed_context = self._compress_context(
            prioritized_content + tables_content, 
            config.max_context_tokens
        )
        
        return {
            'content': compressed_context,
            'tables': tables_content,
            'sources': source_metadata,
            'summary': self._create_context_summary(compressed_context)
        }
    
    def _compress_context(self, 
                         content_list: List[Dict], 
                         max_tokens: int) -> List[Dict]:
        """Compress context to fit within token limits"""
        
        # Simple token estimation (roughly 4 characters per token)
        current_tokens = 0
        compressed = []
        
        for item in content_list:
            estimated_tokens = len(item['text']) // 4
            
            if current_tokens + estimated_tokens <= max_tokens:
                compressed.append(item)
                current_tokens += estimated_tokens
            else:
                # Truncate the last item if it's high relevance
                if item['score'] > 0.7:
                    remaining_tokens = max_tokens - current_tokens
                    truncated_chars = remaining_tokens * 4
                    if truncated_chars > 100:  # Worth including if more than 100 chars
                        item['text'] = item['text'][:truncated_chars] + "..."
                        compressed.append(item)
                break
        
        return compressed
    
    def _create_context_summary(self, content: List[Dict]) -> str:
        """Create a brief summary of the context used"""
        if not content:
            return "No relevant content found"
        
        source_count = len(set(item['source'] for item in content))
        total_chunks = len(content)
        avg_relevance = sum(item['score'] for item in content) / total_chunks
        
        return (f"Analyzed {total_chunks} content chunks from {source_count} sources "
                f"with average relevance score of {avg_relevance:.2f}")
    
    async def _generate_fast_summary(self,
                                   query: str,
                                   context: Dict[str, Any],
                                   config: SummarizationConfig) -> Dict[str, Any]:
        """Generate fast summary using templates and simple processing"""
        
        if not context['content']:
            return self._create_no_context_response()
        
        # Use template-based approach for speed
        if config.response_style == ResponseStyle.FACTUAL:
            template = self._get_factual_template()
        elif config.response_style == ResponseStyle.COMPARATIVE:
            template = self._get_comparative_template()
        else:
            template = self._get_conversational_template()
        
        # Fill template with context
        primary_content = context['content'][0]['text'][:300] if context['content'] else ""
        
        response_text = template.format(
            query=query,
            content=primary_content,
            source=context['content'][0]['source'] if context['content'] else "Unknown"
        )
        
        return {
            'text': response_text,
            'confidence': 0.7,
            'token_count': len(response_text) // 4,
            'coherence_score': 0.8,
            'factual_score': 0.75,
            'style_score': 0.85,
            'reasoning': ["Used template-based generation", "Applied single-source context"],
            'sources': [item['source'] for item in context['content'][:3]],
            'citations': self._create_simple_citations(context['content'][:3])
        }
    
    async def _generate_balanced_summary(self,
                                       query: str,
                                       query_analysis: QueryAnalysis,
                                       context: Dict[str, Any],
                                       config: SummarizationConfig) -> Dict[str, Any]:
        """Generate balanced quality summary with moderate processing"""
        
        if not context['content']:
            return self._create_no_context_response()
        
        # Use LLaMA model if available, otherwise use enhanced template
        if self.model and self.tokenizer:
            return await self._generate_llama_response(query, context, config, "balanced")
        else:
            return await self._generate_enhanced_template_response(
                query, query_analysis, context, config
            )
    
    async def _generate_comprehensive_summary(self,
                                            query: str,
                                            query_analysis: QueryAnalysis,
                                            context: Dict[str, Any],
                                            config: SummarizationConfig) -> Dict[str, Any]:
        """Generate comprehensive summary with full analysis"""
        
        if not context['content']:
            return self._create_no_context_response()
        
        # Use LLaMA model for best quality
        if self.model and self.tokenizer:
            return await self._generate_llama_response(query, context, config, "comprehensive")
        else:
            # Fallback to enhanced processing
            return await self._generate_comprehensive_template_response(
                query, query_analysis, context, config
            )
    
    async def _generate_llama_response(self,
                                     query: str,
                                     context: Dict[str, Any],
                                     config: SummarizationConfig,
                                     quality_mode: str) -> Dict[str, Any]:
        """Generate response using LLaMA model"""
        
        # Build comprehensive prompt
        prompt = self._build_llama_prompt(query, context, config, quality_mode)
        
        # Tokenize and generate
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + config.max_response_length,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response_tokens = outputs[0][inputs.shape[1]:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Calculate quality metrics
        quality_scores = self._calculate_response_quality(response_text, context, query)
        
        return {
            'text': response_text.strip(),
            'confidence': quality_scores['confidence'],
            'token_count': len(response_tokens),
            'coherence_score': quality_scores['coherence'],
            'factual_score': quality_scores['factual'],
            'style_score': quality_scores['style'],
            'reasoning': [
                "Used LLaMA model for generation",
                f"Applied {quality_mode} quality mode",
                f"Processed {len(context['content'])} context chunks"
            ],
            'sources': [item['source'] for item in context['content']],
            'citations': self._create_detailed_citations(context['content'])
        }
    
    async def _generate_enhanced_template_response(self,
                                                 query: str,
                                                 query_analysis: QueryAnalysis,
                                                 context: Dict[str, Any],
                                                 config: SummarizationConfig) -> Dict[str, Any]:
        """Generate enhanced response using intelligent templates (LLaMA fallback)"""
        
        # Multi-source synthesis
        content_synthesis = self._synthesize_multiple_sources(context['content'])
        
        # Intent-specific processing
        if query_analysis.intent.name == "FACTUAL":
            response_text = self._generate_factual_response(query, content_synthesis, context)
        elif query_analysis.intent.name == "ANALYTICAL":
            response_text = self._generate_analytical_response(query, content_synthesis, context)
        elif query_analysis.intent.name == "COMPARATIVE":
            response_text = self._generate_comparative_response(query, content_synthesis, context)
        else:
            response_text = self._generate_conversational_response(query, content_synthesis, context)
        
        # Add entity-specific enhancements
        enhanced_response = self._enhance_with_entities(
            response_text, query_analysis.entities, context
        )
        
        return {
            'text': enhanced_response,
            'confidence': 0.85,
            'token_count': len(enhanced_response) // 4,
            'coherence_score': 0.88,
            'factual_score': 0.82,
            'style_score': 0.90,
            'reasoning': [
                "Used enhanced template processing",
                f"Applied {query_analysis.intent.name} intent adaptation",
                "Synthesized multiple sources",
                f"Enhanced with {len(query_analysis.entities)} entities"
            ],
            'sources': [item['source'] for item in context['content']],
            'citations': self._create_detailed_citations(context['content'])
        }
    
    def _build_llama_prompt(self,
                          query: str,
                          context: Dict[str, Any],
                          config: SummarizationConfig,
                          quality_mode: str) -> str:
        """Build comprehensive prompt for LLaMA model"""
        
        # System instruction
        system_prompt = f"""You are an intelligent assistant that provides accurate, well-structured responses based on provided context. 

Response Style: {config.response_style.value}
Quality Mode: {quality_mode}
Include Citations: {config.include_citations}

Guidelines:
- Base your response strictly on the provided context
- Be accurate and factual
- Match the requested response style
- Include source references when enabled
- Keep response under {config.max_response_length} words
"""
        
        # Context section
        context_text = "\n\n".join([
            f"Source: {item['source']} (Page {item.get('page', 'N/A')}, Relevance: {item['score']:.2f})\n{item['text']}"
            for item in context['content'][:5]  # Top 5 sources
        ])
        
        # User query
        user_prompt = f"""Context:\n{context_text}\n\nQuery: {query}\n\nResponse:"""
        
        return f"{system_prompt}\n\n{user_prompt}"
    
    def _synthesize_multiple_sources(self, content_list: List[Dict]) -> Dict[str, Any]:
        """Synthesize information from multiple sources"""
        
        if not content_list:
            return {'main_points': [], 'supporting_facts': [], 'consensus': 'No content available'}
        
        # Extract key information
        all_text = " ".join([item['text'] for item in content_list])
        source_count = len(set(item['source'] for item in content_list))
        
        # Simple key point extraction (in production, would use more sophisticated NLP)
        sentences = all_text.split('. ')
        key_sentences = [s for s in sentences if len(s) > 50 and any(keyword in s.lower() 
                        for keyword in ['budget', 'cost', 'amount', 'total', 'revenue', 'department'])]
        
        return {
            'main_points': key_sentences[:5],
            'supporting_facts': [item['text'][:200] for item in content_list[:3]],
            'consensus': f"Information synthesized from {source_count} sources",
            'confidence': sum(item['score'] for item in content_list) / len(content_list)
        }
    
    def _generate_factual_response(self, query: str, synthesis: Dict, context: Dict) -> str:
        """Generate factual response with direct answers"""
        
        if not synthesis['main_points']:
            return f"Based on the available documents, I could not find specific factual information to answer: {query}"
        
        response = f"Based on the available documents:\n\n"
        
        # Add main facts
        for i, point in enumerate(synthesis['main_points'][:3], 1):
            response += f"{i}. {point.strip()}.\n"
        
        # Add source confidence
        response += f"\nThis information is based on {len(context['content'])} relevant sources "
        response += f"with an average confidence score of {synthesis['confidence']:.2f}."
        
        return response
    
    def _generate_analytical_response(self, query: str, synthesis: Dict, context: Dict) -> str:
        """Generate analytical response with explanations"""
        
        response = f"To answer your question about {query}, here's the analysis:\n\n"
        
        # Main analysis
        response += "Key Findings:\n"
        for point in synthesis['main_points'][:3]:
            response += f"• {point.strip()}\n"
        
        response += "\nSupporting Evidence:\n"
        for fact in synthesis['supporting_facts']:
            response += f"• {fact}...\n"
        
        response += f"\nConclusion: {synthesis['consensus']}"
        
        return response
    
    def _generate_comparative_response(self, query: str, synthesis: Dict, context: Dict) -> str:
        """Generate comparative response showing differences/similarities"""
        
        response = f"Comparison analysis for: {query}\n\n"
        
        # Look for comparison indicators in content
        comparison_content = []
        for item in context['content']:
            if any(word in item['text'].lower() for word in ['vs', 'versus', 'compared', 'higher', 'lower', 'more', 'less']):
                comparison_content.append(item['text'][:200])
        
        if comparison_content:
            response += "Key Comparisons Found:\n"
            for i, comp in enumerate(comparison_content[:3], 1):
                response += f"{i}. {comp}...\n"
        else:
            response += "Direct comparisons:\n"
            for point in synthesis['main_points'][:3]:
                response += f"• {point}\n"
        
        return response
    
    def _generate_conversational_response(self, query: str, synthesis: Dict, context: Dict) -> str:
        """Generate conversational response in natural language"""
        
        response = f"I can help you with that question about {query.lower()}.\n\n"
        
        if synthesis['main_points']:
            response += "From what I found in the documents, "
            response += synthesis['main_points'][0].lower()
            
            if len(synthesis['main_points']) > 1:
                response += f" Additionally, {synthesis['main_points'][1].lower()}"
        
        response += f"\n\nThis information comes from {len(context['sources'])} different sources "
        response += "in the document collection."
        
        return response
    
    def _enhance_with_entities(self, 
                              response_text: str, 
                              entities: List[Any], 
                              context: Dict) -> str:
        """Enhance response with entity-specific information"""
        
        if not entities:
            return response_text
        
        # Add entity context if relevant
        entity_enhancements = []
        
        for entity in entities[:3]:  # Top 3 entities
            entity_text = entity.text.lower()
            
            # Find context chunks that mention this entity
            relevant_chunks = [
                item for item in context['content'] 
                if entity_text in item['text'].lower()
            ]
            
            if relevant_chunks:
                entity_enhancements.append(
                    f"Regarding {entity.text}: {relevant_chunks[0]['text'][:100]}..."
                )
        
        if entity_enhancements:
            response_text += "\n\nAdditional context:\n"
            for enhancement in entity_enhancements:
                response_text += f"• {enhancement}\n"
        
        return response_text
    
    def _calculate_response_quality(self, 
                                  response_text: str, 
                                  context: Dict, 
                                  query: str) -> Dict[str, float]:
        """Calculate quality metrics for generated response"""
        
        # Simple quality estimation (would be more sophisticated in production)
        
        # Confidence based on context usage
        context_words = set(word.lower() for item in context['content'] for word in item['text'].split())
        response_words = set(response_text.lower().split())
        overlap_ratio = len(context_words.intersection(response_words)) / max(len(response_words), 1)
        
        confidence = min(0.9, overlap_ratio * 1.2)
        
        # Coherence based on structure
        sentences = response_text.split('. ')
        coherence = min(0.95, len(sentences) * 0.1 + 0.7)  # Longer responses tend to be more coherent
        
        # Factual consistency (simplified)
        factual = 0.8 if len(response_text) > 100 else 0.6
        
        # Style alignment (based on length and structure)
        style = 0.85 if 50 <= len(response_text) <= 800 else 0.7
        
        return {
            'confidence': confidence,
            'coherence': coherence,
            'factual': factual,
            'style': style
        }
    
    async def _post_process_response(self,
                                   response: Dict[str, Any],
                                   chunks: List[RankedChunk],
                                   config: SummarizationConfig) -> Dict[str, Any]:
        """Post-process response with citations and quality checks"""
        
        processed_response = response.copy()
        
        # Add citations if enabled
        if config.include_citations:
            processed_response['citations'] = self._create_detailed_citations(chunks)
            
            # Add inline citations if requested
            if config.citation_style == "inline":
                processed_response['text'] = self._add_inline_citations(
                    processed_response['text'], processed_response['citations']
                )
        
        # Quality checks
        if config.enable_fact_checking:
            fact_check_score = self._perform_fact_check(processed_response['text'], chunks)
            processed_response['factual_score'] = min(processed_response['factual_score'], fact_check_score)
        
        if config.enable_coherence_check:
            coherence_score = self._check_coherence(processed_response['text'])
            processed_response['coherence_score'] = min(processed_response['coherence_score'], coherence_score)
        
        return processed_response
    
    def _create_detailed_citations(self, content_or_chunks: Union[List[Dict], List[RankedChunk]]) -> List[Dict[str, Any]]:
        """Create detailed citations from content or chunks"""
        citations = []
        
        for i, item in enumerate(content_or_chunks[:5], 1):  # Top 5 sources
            if isinstance(item, dict):  # Content dict
                citation = {
                    'id': i,
                    'source': item.get('source', 'Unknown'),
                    'page': item.get('page', 'N/A'),
                    'relevance_score': item.get('score', 0.0),
                    'text_snippet': item.get('text', '')[:150] + "..."
                }
            else:  # RankedChunk
                citation = {
                    'id': i,
                    'source': item.chunk.metadata.get('source', 'Unknown'),
                    'page': item.chunk.metadata.get('page_number', 'N/A'),
                    'relevance_score': item.final_rank_score,
                    'text_snippet': item.chunk.content[:150] + "..."
                }
            
            citations.append(citation)
        
        return citations
    
    def _create_simple_citations(self, content_list: List[Dict]) -> List[Dict[str, Any]]:
        """Create simple citations for fast mode"""
        return [
            {
                'id': i,
                'source': item['source'],
                'relevance': item['score']
            }
            for i, item in enumerate(content_list, 1)
        ]
    
    def _add_inline_citations(self, text: str, citations: List[Dict]) -> str:
        """Add inline citations to text"""
        if not citations:
            return text
        
        # Simple approach: add citation at the end
        citation_refs = ", ".join([f"[{cite['id']}]" for cite in citations[:3]])
        return f"{text} {citation_refs}"
    
    def _perform_fact_check(self, response_text: str, chunks: List[RankedChunk]) -> float:
        """Perform simple fact checking against source content"""
        # Simplified fact-checking - compare response against source content
        source_text = " ".join([chunk.chunk.content for chunk in chunks[:3]])
        
        response_words = set(response_text.lower().split())
        source_words = set(source_text.lower().split())
        
        # Calculate overlap ratio as proxy for factual accuracy
        overlap = len(response_words.intersection(source_words))
        total_response_words = len(response_words)
        
        if total_response_words == 0:
            return 0.5
        
        fact_score = min(0.95, (overlap / total_response_words) * 2)
        return fact_score
    
    def _check_coherence(self, text: str) -> float:
        """Check response coherence"""
        sentences = text.split('. ')
        
        if len(sentences) < 2:
            return 0.8
        
        # Simple coherence check based on sentence transitions
        coherent_transitions = 0
        for i in range(len(sentences) - 1):
            current_words = set(sentences[i].lower().split())
            next_words = set(sentences[i + 1].lower().split())
            
            # Check for word overlap between consecutive sentences
            if len(current_words.intersection(next_words)) > 0:
                coherent_transitions += 1
        
        coherence_ratio = coherent_transitions / max(len(sentences) - 1, 1)
        return min(0.95, coherence_ratio + 0.6)
    
    def _get_factual_template(self) -> str:
        """Get template for factual responses"""
        return "Based on the documents, {content} (Source: {source})"
    
    def _get_comparative_template(self) -> str:
        """Get template for comparative responses"""
        return "Comparing the available information: {content} (Source: {source})"
    
    def _get_conversational_template(self) -> str:
        """Get template for conversational responses"""
        return "I can help you with that. {content} This information comes from {source}."
    
    def _create_no_context_response(self) -> Dict[str, Any]:
        """Create response when no relevant context is available"""
        return {
            'text': "I don't have sufficient information in the available documents to answer this question accurately.",
            'confidence': 0.1,
            'token_count': 20,
            'coherence_score': 0.9,
            'factual_score': 0.3,
            'style_score': 0.8,
            'reasoning': ["No relevant context found", "Generated fallback response"],
            'sources': [],
            'citations': []
        }
    
    def _create_error_response(self, error_message: str, processing_time: float) -> SummarizedResponse:
        """Create error response"""
        return SummarizedResponse(
            response_text=f"I encountered an error while processing your request: {error_message}",
            confidence_score=0.0,
            sources_used=[],
            citations=[],
            processing_time_ms=processing_time * 1000,
            tokens_generated=0,
            context_chunks_used=0,
            coherence_score=0.0,
            factual_consistency_score=0.0,
            style_alignment_score=0.0,
            reasoning_steps=[f"Error occurred: {error_message}"],
            context_summary="Processing failed"
        )
    
    def _update_metrics(self, processing_time: float):
        """Update performance metrics"""
        self.request_count += 1
        self.total_processing_time += processing_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if self.request_count == 0:
            return {
                'total_requests': 0,
                'average_processing_time_ms': 0.0,
                'cache_hit_rate': 0.0,
                'model_loaded': self.model is not None
            }
        
        return {
            'total_requests': self.request_count,
            'average_processing_time_ms': self.total_processing_time / self.request_count,
            'cache_hit_rate': len(self.cache) / self.request_count if self.request_count > 0 else 0.0,
            'model_loaded': self.model is not None,
            'device': self.device,
            'cache_size': len(self.cache)
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self.cache.clear()
        self.logger.info("Response cache cleared")
    
    async def _generate_streaming_response(self,
                                         query: str,
                                         query_analysis: QueryAnalysis,
                                         processed_context: Dict[str, Any],
                                         config: SummarizationConfig,
                                         retrieval_result: 'RetrievalResult') -> AsyncGenerator[StreamChunk, None]:
        """
        Generate streaming response for real-time interaction
        
        Yields StreamChunk objects with partial content for smooth user experience
        """
        import uuid
        
        stream_id = str(uuid.uuid4())
        chunk_id = 0
        
        try:
            # Send initial metadata chunk
            yield StreamChunk(
                chunk_id=chunk_id,
                content="",
                metadata={
                    "type": "initialization",
                    "stream_id": stream_id,
                    "query": query,
                    "context_chunks": len(processed_context.get('content', [])),
                    "processing_mode": config.mode.value,
                    "sources": [item['source'] for item in processed_context.get('content', [])[:3]]
                }
            )
            chunk_id += 1
            
            # Generate response based on mode
            if config.mode == SummarizationMode.FAST:
                async for chunk in self._stream_fast_response(query, processed_context, config, chunk_id):
                    yield chunk
                    chunk_id += 1
            elif config.mode == SummarizationMode.COMPREHENSIVE:
                async for chunk in self._stream_comprehensive_response(
                    query, query_analysis, processed_context, config, chunk_id
                ):
                    yield chunk
                    chunk_id += 1
            else:  # BALANCED
                async for chunk in self._stream_balanced_response(
                    query, query_analysis, processed_context, config, chunk_id
                ):
                    yield chunk
                    chunk_id += 1
            
            # Send final metadata chunk
            yield StreamChunk(
                chunk_id=chunk_id,
                content="",
                is_final=True,
                metadata={
                    "type": "completion",
                    "stream_id": stream_id,
                    "total_chunks": chunk_id,
                    "sources_used": [item['source'] for item in processed_context.get('content', [])],
                    "processing_time_ms": time.time() * 1000  # Approximate
                }
            )
            
        except Exception as e:
            # Send error chunk
            yield StreamChunk(
                chunk_id=chunk_id,
                content=f"Error: {str(e)}",
                is_final=True,
                metadata={
                    "type": "error",
                    "stream_id": stream_id,
                    "error": str(e)
                }
            )
    
    async def _stream_fast_response(self,
                                   query: str,
                                   context: Dict[str, Any],
                                   config: SummarizationConfig,
                                   start_chunk_id: int) -> AsyncGenerator[StreamChunk, None]:
        """Stream fast template-based response"""
        
        if not context['content']:
            yield StreamChunk(
                chunk_id=start_chunk_id,
                content="I don't have enough context to answer your question.",
                is_final=True
            )
            return
        
        # Generate template response
        template_response = await self._generate_fast_summary(query, context, config)
        response_text = template_response['text']
        
        # Stream in chunks
        words = response_text.split()
        chunk_size = config.stream_chunk_size
        
        chunk_id = start_chunk_id
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " ".join(chunk_words)
            
            # Add appropriate spacing
            if i > 0:
                chunk_content = " " + chunk_content
            
            yield StreamChunk(
                chunk_id=chunk_id,
                content=chunk_content,
                metadata={
                    "type": "content",
                    "progress": min((i + chunk_size) / len(words), 1.0),
                    "confidence": template_response['confidence']
                }
            )
            
            chunk_id += 1
            
            # Streaming delay for smooth experience
            if config.stream_delay_ms > 0:
                await asyncio.sleep(config.stream_delay_ms / 1000.0)
    
    async def _stream_balanced_response(self,
                                       query: str,
                                       query_analysis: QueryAnalysis,
                                       context: Dict[str, Any],
                                       config: SummarizationConfig,
                                       start_chunk_id: int) -> AsyncGenerator[StreamChunk, None]:
        """Stream balanced quality response with LLaMA or enhanced templates"""
        
        if self.model and self.tokenizer:
            # Use LLaMA streaming if available
            async for chunk in self._stream_llama_response(query, context, config, start_chunk_id, "balanced"):
                yield chunk
        else:
            # Use enhanced template with streaming
            enhanced_response = await self._generate_enhanced_template_response(
                query, query_analysis, context, config
            )
            
            # Stream the enhanced response
            response_text = enhanced_response['text']
            words = response_text.split()
            chunk_size = config.stream_chunk_size
            
            chunk_id = start_chunk_id
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_content = " ".join(chunk_words)
                
                if i > 0:
                    chunk_content = " " + chunk_content
                
                yield StreamChunk(
                    chunk_id=chunk_id,
                    content=chunk_content,
                    metadata={
                        "type": "content",
                        "progress": min((i + chunk_size) / len(words), 1.0),
                        "confidence": enhanced_response['confidence']
                    }
                )
                
                chunk_id += 1
                
                if config.stream_delay_ms > 0:
                    await asyncio.sleep(config.stream_delay_ms / 1000.0)
    
    async def _stream_comprehensive_response(self,
                                           query: str,
                                           query_analysis: QueryAnalysis,
                                           context: Dict[str, Any],
                                           config: SummarizationConfig,
                                           start_chunk_id: int) -> AsyncGenerator[StreamChunk, None]:
        """Stream comprehensive quality response"""
        
        if self.model and self.tokenizer:
            # Use LLaMA streaming for best quality
            async for chunk in self._stream_llama_response(query, context, config, start_chunk_id, "comprehensive"):
                yield chunk
        else:
            # Use comprehensive template with streaming
            comprehensive_response = await self._generate_comprehensive_template_response(
                query, query_analysis, context, config
            )
            
            response_text = comprehensive_response['text']
            words = response_text.split()
            chunk_size = config.stream_chunk_size
            
            chunk_id = start_chunk_id
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_content = " ".join(chunk_words)
                
                if i > 0:
                    chunk_content = " " + chunk_content
                
                yield StreamChunk(
                    chunk_id=chunk_id,
                    content=chunk_content,
                    metadata={
                        "type": "content",
                        "progress": min((i + chunk_size) / len(words), 1.0),
                        "confidence": comprehensive_response['confidence'],
                        "reasoning_step": len([w for w in chunk_words if w.endswith('.')])  # Approximate reasoning steps
                    }
                )
                
                chunk_id += 1
                
                if config.stream_delay_ms > 0:
                    await asyncio.sleep(config.stream_delay_ms / 1000.0)
    
    async def _stream_llama_response(self,
                                   query: str,
                                   context: Dict[str, Any],
                                   config: SummarizationConfig,
                                   start_chunk_id: int,
                                   quality_mode: str) -> AsyncGenerator[StreamChunk, None]:
        """Stream LLaMA model response (mock implementation for demonstration)"""
        
        # Mock streaming implementation - in production, would use actual LLaMA streaming
        mock_responses = {
            "balanced": f"Based on the provided context, here's what I can tell you about {query}: The information suggests several key points that are relevant to your question.",
            "comprehensive": f"To thoroughly address your question about {query}, I need to analyze the context from multiple perspectives. The available information provides several insights that help form a comprehensive response."
        }
        
        response_text = mock_responses.get(quality_mode, mock_responses["balanced"])
        words = response_text.split()
        chunk_size = config.stream_chunk_size
        
        chunk_id = start_chunk_id
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " ".join(chunk_words)
            
            if i > 0:
                chunk_content = " " + chunk_content
            
            yield StreamChunk(
                chunk_id=chunk_id,
                content=chunk_content,
                metadata={
                    "type": "content",
                    "progress": min((i + chunk_size) / len(words), 1.0),
                    "model": "llama_mock",
                    "quality_mode": quality_mode
                }
            )
            
            chunk_id += 1
            
            # Slightly longer delay for LLaMA to simulate processing
            if config.stream_delay_ms > 0:
                await asyncio.sleep((config.stream_delay_ms + 20) / 1000.0)
    
    async def _create_error_stream(self, error_message: str) -> AsyncGenerator[StreamChunk, None]:
        """Create error stream for streaming mode failures"""
        yield StreamChunk(
            chunk_id=0,
            content=f"Error: {error_message}",
            is_final=True,
            metadata={
                "type": "error",
                "error": error_message
            }
        )


# Configuration helpers
def create_fast_summarization_config() -> SummarizationConfig:
    """Create configuration optimized for speed"""
    return SummarizationConfig(
        mode=SummarizationMode.FAST,
        max_response_length=256,
        temperature=0.3,
        max_context_tokens=1024,
        include_citations=False,
        enable_fact_checking=False,
        enable_coherence_check=False
    )


def create_balanced_summarization_config() -> SummarizationConfig:
    """Create balanced configuration (default)"""
    return SummarizationConfig(
        mode=SummarizationMode.BALANCED,
        max_response_length=512,
        temperature=0.7,
        max_context_tokens=2048,
        include_citations=True,
        enable_fact_checking=True,
        enable_coherence_check=True
    )


def create_comprehensive_summarization_config() -> SummarizationConfig:
    """Create configuration optimized for quality"""
    return SummarizationConfig(
        mode=SummarizationMode.COMPREHENSIVE,
        max_response_length=768,
        temperature=0.8,
        max_context_tokens=3072,
        include_citations=True,
        citation_style="inline",
        enable_fact_checking=True,
        enable_coherence_check=True,
        min_confidence_score=0.7
    )


def create_streaming_summarization_config(mode: SummarizationMode = SummarizationMode.BALANCED,
                                        chunk_size: int = 50,
                                        delay_ms: int = 50) -> SummarizationConfig:
    """Create configuration optimized for streaming"""
    config = SummarizationConfig(
        mode=mode,
        enable_streaming=True,
        stream_chunk_size=chunk_size,
        stream_delay_ms=delay_ms
    )
    
    # Optimize for streaming
    if mode == SummarizationMode.FAST:
        config.max_response_length = 256
        config.temperature = 0.3
        config.stream_chunk_size = 30  # Smaller chunks for faster start
        config.stream_delay_ms = 30
    elif mode == SummarizationMode.COMPREHENSIVE:
        config.max_response_length = 768
        config.temperature = 0.8
        config.stream_chunk_size = 70  # Larger chunks for better context
        config.stream_delay_ms = 70
    
    return config
