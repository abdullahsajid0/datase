import json
import pandas as pd
import random
from datetime import datetime
from typing import List, Dict
import re
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from together import api as together  # Assuming 'together' is a package

class QualityMetrics:
    def __init__(self, complexity_score, diversity_score, technical_density, avg_sentence_length, word_count):
        self.complexity_score = complexity_score
        self.diversity_score = diversity_score
        self.technical_density = technical_density
        self.avg_sentence_length = avg_sentence_length
        self.word_count = word_count

class DatasetGenerator:
    def __init__(self, embedding_model, model_name, temperature):
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.temperature = temperature
        self.context_data = ""
    
    def add_context_data(self, source_files):
        """Process uploaded files into context knowledge"""
        st.write("📝 Processing uploaded files...")
        context_data = []
        for file in source_files:
            st.write(f"  - Reading {file.name}")
            content = file.read().decode('utf-8')
            if file.name.endswith('.txt'):
                lines = content.split('\n')
                st.write(f"    Found {len(lines)} lines in txt file")
                context_data.extend(lines)
            elif file.name.endswith('.csv'):
                df = pd.read_csv(pd.StringIO(content))
                st.write(f"    Found {len(df)} rows in csv file")
                context_data.extend(df.iloc[:, 0].tolist())
            elif file.name.endswith(('.json', '.jsonl')):
                lines = [line for line in content.split('\n') if line.strip()]
                st.write(f"    Found {len(lines)} lines in json file")
                for line in lines:
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict) and 'text' in data:
                            context_data.append(data['text'])
                    except json.JSONDecodeError:
                        continue
        
        self.context_data = "\n".join(context_data)
        return len(context_data)
    
    def generate_dataset(self, topic: str, concepts: List[str], num_samples: int) -> List[Dict]:
        """Generate high-quality conversational dataset"""
        st.write(f"🎯 Generating enhanced dataset for topic: {topic}")
        dataset = []
        total_steps = len(concepts) * num_samples
        current_step = 0
        dashboard_placeholder = st.empty()

        for concept in concepts:
            try:
                concept_context = self._get_relevant_context(concept)
                used_prompts = set()

                for i in range(num_samples):
                    try:
                        st.write("      Generating unique prompt...")
                        attempt = 0
                        while attempt < 5:
                            user_query = self._generate_unique_prompt(concept, topic)
                            if user_query not in used_prompts:
                                used_prompts.add(user_query)
                                st.write(f"      Generated prompt: {user_query}")
                                break
                            attempt += 1
                        
                        st.write("      Generating response...")
                        response = self._generate_enhanced_response(
                            user_query=user_query,
                            concept=concept,
                            topic=topic,
                            context=concept_context
                        )
                        
                        if not response:
                            raise ValueError("Empty response generated")
                        
                        st.write(f"      Generated {len(response.split())} words")
                        
                        st.write("      Calculating quality metrics...")
                        metrics = self._calculate_quality_metrics(response)
                        
                        cleaned_response = response.split("Assistant:")[-1].strip()
                        entry = {
                            "prompt": user_query,
                            "response": cleaned_response,
                            "metadata": {
                                "topic": topic,
                                "concept": concept,
                                "model": self.model_name,
                                "has_context": bool(self.context_data),
                                "type": "chat",
                                "quality_metrics": metrics.__dict__,
                                "generation_timestamp": datetime.now().isoformat()
                            }
                        }
                        dataset.append(entry)
                        current_step += 1
                        
                        with dashboard_placeholder:
                            self.create_dashboard(dataset, current_step, total_steps)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating sample {i+1}: {str(e)}")
                        continue
                
            except Exception as e:
                st.error(f"❌ Error processing concept {concept}: {str(e)}")
                continue
        
        if not dataset:
            st.error("❌ No samples were generated successfully")
            raise ValueError("Failed to generate any valid samples")
        
        st.success(f"✅ Generated {len(dataset)} samples successfully")
        return dataset

    def _get_relevant_context(self, concept: str) -> str:
        """Get most relevant context using semantic search"""
        if not self.context_data:
            return ""
            
        context_chunks = self.context_data.split('\n')
        
        concept_embedding = self.embedding_model.encode([concept])[0]
        chunk_embeddings = self.embedding_model.encode(context_chunks)
        
        similarities = cosine_similarity([concept_embedding], chunk_embeddings)[0]
        top_indices = similarities.argsort()[-3:][::-1]
        relevant_chunks = [context_chunks[i] for i in top_indices]
        
        return "\n".join(relevant_chunks)

    def _generate_enhanced_response(self, user_query: str, concept: str, topic: str, context: str) -> str:
        """Generate response using Together AI API"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"You are an expert in {topic}, specifically about {concept}. Use this context in your response: {context}"
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ]
            
            together.api_key = "b2934a0d84d45a7511b9e1e9f62db2f6d2a7e388521f367ea1d2de47635ad201"
            client = together.Together(api_key=together.api_key)
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=512,
                temperature=self.temperature,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1.2,
                stop=["<|eot_id|>", "<|eom_id|>"],
                stream=True
            )
            
            generated_text = ""
            for token in response:
                if hasattr(token, 'choices'):
                    chunk = token.choices[0].delta.content
                    if chunk:
                        generated_text += chunk
            
            if not generated_text.strip():
                raise ValueError("No response generated")
            
            return generated_text.strip()
            
        except Exception as e:
            st.error(f"❌ Error in response generation: {str(e)}")
            raise

    def _calculate_quality_metrics(self, text: str) -> QualityMetrics:
        """Calculate quality metrics for generated text"""
        words = text.split()
        sentences = text.split('.')
        technical_terms = len(re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', text))
        
        return QualityMetrics(
            complexity_score=sum(len(word) for word in words) / len(words),
            diversity_score=len(set(words)) / len(words),
            technical_density=technical_terms / len(words),
            avg_sentence_length=len(words) / len(sentences),
            word_count=len(words)
        )

    def _generate_unique_prompt(self, concept: str, topic: str) -> str:
        """Generate unique, contextual prompts"""
        templates = [
            f"How can I get started with {concept} in {topic}?",
            f"What are the fundamental concepts of {concept}?",
            f"Could you explain {concept} in simple terms?",
            f"What's the best way to implement {concept}?",
            f"Can you show me a practical example of {concept}?",
            f"What are common mistakes to avoid with {concept}?",
            f"How can I optimize {concept} in my project?",
            f"What are the best practices for {concept}?",
            f"How does {concept} integrate with other tools?"
        ]
        
        prefixes = ["", "Hey! ", "Quick question: ", "I need help: "]
        suffixes = ["", " Any suggestions?", " Thanks in advance!"]
        
        template = random.choice(templates)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        
        return f"{prefix}{template}{suffix}".strip()

    def create_dashboard(self, dataset: List[Dict], current_step: int, total_steps: int):
        """Create real-time dashboard with charts and animations"""
        dashboard = st.container()
        with dashboard:
            st.markdown("## 📊 Generation Dashboard")
            progress_percentage = (current_step / total_steps) * 100
            st.progress(progress_percentage / 100)
            st.metric("Progress", f"{progress_percentage:.1f}%", f"{current_step}/{total_steps} rows")
            
            if dataset:
                history_df = pd.DataFrame([
                    {
                        'index': i,
                        'complexity_score': d['metadata']['quality_metrics']['complexity_score'],
                        'diversity_score': d['metadata']['quality_metrics']['diversity_score'],
                        'technical_density': d['metadata']['quality_metrics']['technical_density'],
                        'avg_sentence_length': d['metadata']['quality_metrics']['avg_sentence_length'],
                        'word_count': d['metadata']['quality_metrics']['word_count']
                    } for i, d in enumerate(dataset)
                ])
                
                st.write("Metrics Summary", history_df.describe())
                
                # Create quality metrics chart
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(history_df['index'], history_df['complexity_score'], label="Complexity Score")
                ax.bar(history_df['index'], history_df['diversity_score'], label="Diversity Score", alpha=0.7)
                ax.set_title("Quality Metrics")
                ax.legend()
                st.pyplot(fig)

