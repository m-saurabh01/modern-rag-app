#!/usr/bin/env python3
"""
QueryAnalyzer Demonstration Script

This script demonstrates the capabilities of the QueryAnalyzer service,
showcasing intelligent query understanding, entity extraction, and 
retrieval optimization for the Modern RAG system.

Usage:
    python demo_query_analyzer.py

Key Features Demonstrated:
- Query intent classification (6 types)
- Entity extraction with confidence scoring
- Conservative query expansion for precision
- Retrieval strategy suggestions
- Performance monitoring
- Caching capabilities
"""

import asyncio
import sys
import os
from typing import List, Dict, Any
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our services
from services.query_analyzer import QueryAnalyzer
from models.query_models import (
    QueryAnalyzerConfig, QueryCache, ProcessingMode,
    QueryIntent, QuestionType, EntityType, RetrievalStrategy
)
from config.settings import Settings


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_analysis_summary(analysis, query_num: int = None):
    """Print a comprehensive analysis summary."""
    prefix = f"Query {query_num}: " if query_num else ""
    
    print(f"\n🔍 {prefix}ANALYSIS RESULTS")
    print(f"   Original Query: '{analysis.original_query}'")
    print(f"   Intent: {analysis.intent.value.upper()} (confidence: {analysis.confidence_score:.2f})")
    print(f"   Question Type: {analysis.question_type.value.upper()}")
    print(f"   Complexity Score: {analysis.complexity_score:.2f}")
    print(f"   Processing Time: {analysis.processing_time:.1f}ms")
    
    # Entities
    if analysis.entities:
        print(f"\n📍 EXTRACTED ENTITIES ({len(analysis.entities)}):")
        for entity in analysis.entities:
            print(f"   • {entity.text} ({entity.entity_type.value}) - confidence: {entity.confidence:.2f}")
    else:
        print(f"\n📍 No entities extracted")
    
    # Query Expansion
    if analysis.expansion.synonyms or analysis.expansion.domain_terms:
        print(f"\n🔄 QUERY EXPANSION (confidence: {analysis.expansion.expansion_confidence:.2f}):")
        
        if analysis.expansion.synonyms:
            print("   Synonyms:")
            for term, synonyms in analysis.expansion.synonyms.items():
                print(f"     {term}: {', '.join(synonyms[:3])}")
        
        if analysis.expansion.domain_terms:
            print(f"   Domain Terms: {', '.join(analysis.expansion.domain_terms[:5])}")
        
        if analysis.expansion.suggested_filters:
            print(f"   Suggested Filters: {analysis.expansion.suggested_filters}")
    
    # Retrieval Strategies
    if analysis.suggested_strategies:
        print(f"\n🎯 SUGGESTED RETRIEVAL STRATEGIES:")
        for strategy in analysis.suggested_strategies:
            print(f"   • {strategy.value.replace('_', ' ').title()}")
    
    # Performance Metadata
    nlp_status = analysis.metadata.get('nlp_libraries_used', {})
    print(f"\n⚙️  PROCESSING DETAILS:")
    print(f"   Mode: {analysis.processing_mode.value}")
    print(f"   NLTK Available: {nlp_status.get('nltk', 'Unknown')}")
    print(f"   spaCy Available: {nlp_status.get('spacy', 'Unknown')}")


async def demo_basic_functionality():
    """Demonstrate basic QueryAnalyzer functionality."""
    print_header("BASIC FUNCTIONALITY DEMONSTRATION")
    
    # Create analyzer with default configuration
    analyzer = QueryAnalyzer()
    
    # Sample queries representing different intents and complexities
    demo_queries = [
        "What is the IT budget allocation for 2024?",
        "Why did the costs increase in Q2?", 
        "Compare Q1 vs Q2 performance for Department of Health",
        "How do I submit a change request?",
        "Is Policy HHS-2024-001 still active?",
        "Tell me about recent department restructuring"
    ]
    
    print("🚀 Analyzing sample queries with different intents...")
    
    for i, query in enumerate(demo_queries, 1):
        analysis = await analyzer.analyze_query(query)
        print_analysis_summary(analysis, i)
    
    return analyzer


async def demo_processing_modes():
    """Demonstrate different processing modes."""
    print_header("PROCESSING MODES COMPARISON")
    
    test_query = "Department of Health budget analysis for Q2 2024 with focus on equipment purchases"
    
    modes = [
        (ProcessingMode.FAST, "Pattern-based only"),
        (ProcessingMode.BALANCED, "Mixed NLP + patterns (DEFAULT)"),
        (ProcessingMode.COMPREHENSIVE, "Full NLP analysis")
    ]
    
    print(f"🎛️  Testing query: '{test_query}'")
    
    for mode, description in modes:
        config = QueryAnalyzerConfig(processing_mode=mode)
        analyzer = QueryAnalyzer(config=config)
        
        analysis = await analyzer.analyze_query(test_query)
        
        print(f"\n📊 {mode.value.upper()} MODE ({description}):")
        print(f"   Processing Time: {analysis.processing_time:.1f}ms")
        print(f"   Entities Found: {len(analysis.entities)}")
        print(f"   Confidence: {analysis.confidence_score:.2f}")
        print(f"   Intent: {analysis.intent.value}")


async def demo_entity_extraction():
    """Demonstrate entity extraction capabilities."""
    print_header("ENTITY EXTRACTION CAPABILITIES")
    
    analyzer = QueryAnalyzer()
    
    # Queries designed to test different entity types
    entity_test_queries = [
        "Find documents from Department of Health dated January 2024",
        "What was the budget allocation of $1,500,000 for IT equipment?",
        "Contact Sarah Johnson at sarah.johnson@health.gov about Policy HHS-2024-001",
        "Compare Q1 2024 vs Q2 2024 performance metrics",
        "Submit change request CR-123 to Finance Department by March 31st"
    ]
    
    print("🏷️  Testing entity extraction across different types...")
    
    for i, query in enumerate(entity_test_queries, 1):
        analysis = await analyzer.analyze_query(query)
        
        print(f"\nQuery {i}: '{query}'")
        
        if analysis.entities:
            entities_by_type = {}
            for entity in analysis.entities:
                if entity.entity_type not in entities_by_type:
                    entities_by_type[entity.entity_type] = []
                entities_by_type[entity.entity_type].append(entity)
            
            for entity_type, entities in entities_by_type.items():
                print(f"   {entity_type.value.upper()}: ", end="")
                entity_texts = [f"{e.text} ({e.confidence:.2f})" for e in entities]
                print(f"{', '.join(entity_texts)}")
        else:
            print("   No entities extracted")


async def demo_query_expansion():
    """Demonstrate conservative query expansion."""
    print_header("CONSERVATIVE QUERY EXPANSION")
    
    # Configure for expansion demonstration
    config = QueryAnalyzerConfig(
        enable_query_expansion=True,
        expansion_aggressiveness=0.3,  # Conservative
    )
    analyzer = QueryAnalyzer(config=config)
    
    expansion_test_queries = [
        "budget allocation report",
        "employee training program",
        "policy implementation guidelines",
        "quarterly performance analysis"
    ]
    
    print("🔄 Testing conservative query expansion...")
    
    for query in expansion_test_queries:
        analysis = await analyzer.analyze_query(query)
        expansion = analysis.expansion
        
        print(f"\n📝 Query: '{query}'")
        print(f"   Original Terms: {expansion.original_terms}")
        
        if expansion.synonyms:
            print("   Synonyms:")
            for term, synonyms in expansion.synonyms.items():
                print(f"     {term}: {synonyms}")
        
        print(f"   Expansion Confidence: {expansion.expansion_confidence:.2f}")


async def demo_caching_performance():
    """Demonstrate caching functionality and performance."""
    print_header("CACHING PERFORMANCE DEMONSTRATION")
    
    # Create analyzer with caching enabled
    cache_config = QueryCache(
        enabled=True,
        ttl_seconds=3600,
        max_entries=100
    )
    config = QueryAnalyzerConfig(cache_config=cache_config)
    analyzer = QueryAnalyzer(config=config)
    
    test_query = "What is the Department of Health budget for 2024?"
    
    print("🔄 Testing caching performance...")
    
    # First run - cache miss
    start_time = datetime.now()
    analysis1 = await analyzer.analyze_query(test_query)
    first_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # Second run - potential cache hit
    start_time = datetime.now() 
    analysis2 = await analyzer.analyze_query(test_query)
    second_time = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"\n⏱️  TIMING RESULTS:")
    print(f"   First analysis: {first_time:.1f}ms")
    print(f"   Second analysis: {second_time:.1f}ms")
    print(f"   Results identical: {analysis1.intent == analysis2.intent}")
    
    # Performance stats
    stats = analyzer.get_performance_stats()
    print(f"\n📊 PERFORMANCE STATISTICS:")
    for key, value in stats.items():
        print(f"   {key}: {value}")


async def demo_batch_processing():
    """Demonstrate batch query processing."""
    print_header("BATCH PROCESSING CAPABILITIES")
    
    analyzer = QueryAnalyzer()
    
    batch_queries = [
        "What is the current budget?",
        "Who is responsible for IT security?", 
        "How much was spent on training?",
        "When is the next review scheduled?",
        "Where can I find the policy documents?"
    ]
    
    print(f"📦 Processing {len(batch_queries)} queries in batch...")
    
    start_time = datetime.now()
    analyses = await analyzer.batch_analyze(batch_queries)
    batch_time = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"   Total batch time: {batch_time:.1f}ms")
    print(f"   Average per query: {batch_time/len(batch_queries):.1f}ms")
    
    # Summary of results
    intent_counts = {}
    for analysis in analyses:
        intent = analysis.intent
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print("\n📊 BATCH RESULTS SUMMARY:")
    for intent, count in intent_counts.items():
        print(f"   {intent.value}: {count} queries")


async def demo_error_handling():
    """Demonstrate error handling and fallback behavior."""
    print_header("ERROR HANDLING & FALLBACK BEHAVIOR")
    
    analyzer = QueryAnalyzer()
    
    # Test various edge cases
    edge_cases = [
        ("", "Empty query"),
        ("???", "Only punctuation"),
        ("a" * 500, "Very long query"),
        ("12345", "Only numbers"),
        ("Special chars: !@#$%^&*()", "Special characters")
    ]
    
    print("🛡️  Testing error handling with edge cases...")
    
    for query, description in edge_cases:
        try:
            if query == "":
                print(f"\n⚠️  {description}: Expecting ValueError...")
                analysis = await analyzer.analyze_query(query)
            else:
                analysis = await analyzer.analyze_query(query)
                print(f"\n✅ {description}: Handled gracefully")
                print(f"   Intent: {analysis.intent.value}")
                print(f"   Confidence: {analysis.confidence_score:.2f}")
        except ValueError as e:
            print(f"   Expected error: {e}")
        except Exception as e:
            print(f"   Unexpected error: {e}")


async def interactive_demo():
    """Interactive demonstration allowing user input."""
    print_header("INTERACTIVE QUERY ANALYSIS")
    
    config = QueryAnalyzerConfig(
        processing_mode=ProcessingMode.BALANCED,
        enable_query_expansion=True,
        expansion_aggressiveness=0.3
    )
    analyzer = QueryAnalyzer(config=config)
    
    print("💬 Interactive Query Analysis Demo")
    print("   Enter your queries to see intelligent analysis")
    print("   Type 'quit' to exit, 'stats' for performance statistics")
    
    query_count = 0
    
    while True:
        try:
            query = input("\n🔍 Enter query: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'stats':
                stats = analyzer.get_performance_stats()
                print("\n📊 Performance Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            elif not query:
                print("⚠️  Please enter a non-empty query")
                continue
            
            query_count += 1
            analysis = await analyzer.analyze_query(query)
            print_analysis_summary(analysis)
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting interactive demo...")
            break
        except Exception as e:
            print(f"❌ Error analyzing query: {e}")
    
    print(f"\n📈 Session completed: {query_count} queries analyzed")


async def main():
    """Main demonstration runner."""
    print("🎯 MODERN RAG APP - QUERY ANALYZER DEMONSTRATION")
    print("   Phase 3.3c: Advanced Query Analysis")
    print("   Intelligent query understanding for optimized retrieval")
    
    print("\n🏗️  INITIALIZATION")
    print("   Setting up QueryAnalyzer with offline NLP capabilities...")
    print("   • Pattern-based fallbacks ensure 100% uptime")
    print("   • Conservative expansion for precision over recall")
    print("   • Integration-ready for DocumentAnalyzer and ChunkingService")
    
    try:
        # Run all demonstrations
        analyzer = await demo_basic_functionality()
        await demo_processing_modes()
        await demo_entity_extraction()
        await demo_query_expansion()
        await demo_caching_performance()
        await demo_batch_processing()
        await demo_error_handling()
        
        # Final summary
        print_header("DEMONSTRATION SUMMARY")
        stats = analyzer.get_performance_stats()
        print("✅ QueryAnalyzer successfully demonstrated all capabilities:")
        print("   • Intent classification across 6 categories")
        print("   • Entity extraction with 13+ entity types")
        print("   • Conservative query expansion for precision")
        print("   • Configurable processing modes (FAST/BALANCED/COMPREHENSIVE)")
        print("   • Optional caching for repeated queries")
        print("   • Comprehensive error handling and fallbacks")
        print("   • Integration-ready architecture")
        
        print(f"\n📊 FINAL STATISTICS:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Option for interactive demo
        response = input("\n🤔 Would you like to try the interactive demo? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            await interactive_demo()
        
        print("\n🎉 PHASE 3.3c IMPLEMENTATION COMPLETE!")
        print("   QueryAnalyzer is ready for integration with Phase 3.3d: Intelligent Retrieval")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
