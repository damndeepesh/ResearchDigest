from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk
from typing import Optional
import re

class PaperSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the paper summarizer with a pre-trained model
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.summarizer = None
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        self._load_model()
    
    def _load_model(self):
        """
        Load the summarization model
        """
        try:
            print(f"Loading summarization model: {self.model_name}")
            
            # Try to use GPU if available
            device = 0 if torch.cuda.is_available() else -1
            
            # Load the summarization pipeline with increased length for more detailed summaries
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                device=device,
                max_length=400,  # Increased from 300 for more detailed summaries
                min_length=150,  # Increased from 100 for more comprehensive content
                do_sample=False,
                temperature=0.7,  # Add some creativity while maintaining accuracy
                top_p=0.9,  # Nucleus sampling for better quality
                repetition_penalty=1.2  # Reduce repetition
            )
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to T5-small model...")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """
        Load a fallback model if the primary model fails
        """
        try:
            fallback_model = "t5-small"
            print(f"Loading fallback model: {fallback_model}")
            
            self.summarizer = pipeline(
                "summarization",
                model=fallback_model,
                max_length=400,  # Increased from 300 for more detailed summaries
                min_length=150,  # Increased from 100 for more comprehensive content
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
            print("Fallback model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            self.summarizer = None
    
    def summarize(self, text: str, max_length: int = 400, min_length: int = 150) -> str:
        """
        Summarize the given text with enhanced quality and detail
        """
        if not text or not text.strip():
            return "No text provided for summarization."
        
        # Check if text is too short for meaningful summarization
        if len(text.strip()) < 100:  # Increased minimum length requirement
            return f"Text too short for meaningful summarization. Original text: {text[:150]}{'...' if len(text) > 150 else ''}"
        
        if not self.summarizer:
            return self._fallback_summarize(text)
        
        try:
            # Clean and preprocess text with enhanced cleaning
            cleaned_text = self._preprocess_text(text)
            
            # Check if text is too short after preprocessing
            if len(cleaned_text.split()) < 30:  # Increased minimum word requirement
                return f"Text too short after preprocessing. Original text: {text[:200]}{'...' if len(text) > 200 else ''}"
            
            # Dynamically adjust summary length based on input text length
            word_count = len(cleaned_text.split())
            if word_count < 100:
                # For very short texts, use shorter summary
                adjusted_max_length = min(max_length, word_count // 2)
                adjusted_min_length = min(min_length, adjusted_max_length // 2)
            elif word_count < 300:
                # For short texts, ensure max_length doesn't exceed input
                adjusted_max_length = min(max_length, word_count // 3)
                adjusted_min_length = min(min_length, adjusted_max_length // 2)
            else:
                # For longer texts, use original parameters
                adjusted_max_length = max_length
                adjusted_min_length = min_length
            
            # Ensure minimum reasonable values
            adjusted_max_length = max(50, adjusted_max_length)
            adjusted_min_length = max(25, adjusted_min_length)
            
            # For very long texts (full papers), use chunking approach
            if word_count > 2000:
                return self._summarize_long_paper(cleaned_text, adjusted_max_length, adjusted_min_length)
            
            # Truncate text if it's too long (models have input length limits)
            max_input_length = 1024
            if word_count > max_input_length:
                # Try to keep the most important parts (beginning and end)
                words = cleaned_text.split()
                if len(words) > max_input_length:
                    # Keep first 60% and last 40% of the text
                    first_part = ' '.join(words[:int(max_input_length * 0.6)])
                    last_part = ' '.join(words[-int(max_input_length * 0.4):])
                    cleaned_text = first_part + " " + last_part
            
            # Generate summary with enhanced parameters
            summary = self.summarizer(
                cleaned_text,
                max_length=adjusted_max_length,
                min_length=adjusted_min_length,
                do_sample=True,  # Enable sampling for better quality
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                truncation=True
            )
            
            if summary and len(summary) > 0:
                summary_text = summary[0]['summary_text']
                if summary_text and len(summary_text.strip()) > 0:
                    # Post-process the summary for better readability
                    enhanced_summary = self._enhance_summary(summary_text, text)
                    return enhanced_summary
                else:
                    return self._fallback_summarize(text)
            else:
                return self._fallback_summarize(text)
                
        except Exception as e:
            print(f"Error in summarization: {e}")
            return self._fallback_summarize(text)
    
    def _summarize_long_paper(self, text: str, max_length: int = 600, min_length: int = 300) -> str:
        """
        Summarize long papers by breaking them into chunks and creating a comprehensive summary
        """
        try:
            print(f"Processing long paper with {len(text.split())} words using chunking approach")
            
            # Split text into logical sections
            sections = self._split_into_sections(text)
            
            if len(sections) <= 1:
                # If we can't split into sections, use simple chunking
                return self._summarize_with_chunks(text, max_length, min_length)
            
            # Summarize each section
            section_summaries = []
            for i, section in enumerate(sections):
                if len(section.strip()) > 100:  # Only summarize substantial sections
                    try:
                        section_summary = self._summarize_section(section, i, len(sections))
                        if section_summary:
                            section_summaries.append(section_summary)
                    except Exception as e:
                        print(f"Error summarizing section {i}: {e}")
                        continue
            
            if not section_summaries:
                # Fallback to chunk-based summarization
                return self._summarize_with_chunks(text, max_length, min_length)
            
            # Combine section summaries
            combined_summary = ' '.join(section_summaries)
            
            # If combined summary is too long, summarize it again
            if len(combined_summary.split()) > max_length * 2:
                final_summary = self.summarizer(
                    combined_summary,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    truncation=True
                )
                
                if final_summary and len(final_summary) > 0:
                    return final_summary[0]['summary_text']
            
            return combined_summary
            
        except Exception as e:
            print(f"Error in long paper summarization: {e}")
            return self._summarize_with_chunks(text, max_length, min_length)
    
    def _split_into_sections(self, text: str) -> list:
        """
        Split text into logical sections based on common paper structure
        """
        try:
            # Common section headers in research papers
            section_patterns = [
                r'\b(?:Abstract|Introduction|Related Work|Methodology|Methods|Approach|Experimental Setup|Experiments|Results|Discussion|Conclusion|Future Work|References?)\b',
                r'\b\d+\.\s+[A-Z][a-zA-Z\s]+\b',  # Numbered sections like "1. Introduction"
                r'\b[A-Z][a-zA-Z\s]+\b'  # Capitalized section headers
            ]
            
            # Find potential section boundaries
            section_boundaries = []
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in section_patterns):
                    if len(line) > 3 and len(line) < 100:  # Reasonable section header length
                        section_boundaries.append(i)
            
            # If we found section boundaries, split accordingly
            if len(section_boundaries) > 1:
                sections = []
                for j in range(len(section_boundaries)):
                    start = section_boundaries[j]
                    end = section_boundaries[j + 1] if j + 1 < len(section_boundaries) else len(lines)
                    
                    section_text = '\n'.join(lines[start:end])
                    if len(section_text.strip()) > 50:  # Only include substantial sections
                        sections.append(section_text)
                
                return sections
            
            # Fallback: split by paragraphs
            paragraphs = text.split('\n\n')
            return [p.strip() for p in paragraphs if len(p.strip()) > 100]
            
        except Exception as e:
            print(f"Error splitting text into sections: {e}")
            # Fallback to simple paragraph splitting
            return text.split('\n\n')
    
    def _summarize_section(self, section_text: str, section_index: int, total_sections: int) -> str:
        """
        Summarize a single section of the paper
        """
        try:
            # Determine section type and adjust summarization accordingly
            section_lower = section_text.lower()
            
            if any(word in section_lower for word in ['abstract', 'introduction']):
                # Abstract and introduction get more detailed summaries
                summary_length = 150
            elif any(word in section_lower for word in ['method', 'approach', 'experiment']):
                # Methods and experiments get medium summaries
                summary_length = 120
            elif any(word in section_lower for word in ['result', 'discussion', 'conclusion']):
                # Results and conclusions get detailed summaries
                summary_length = 150
            else:
                # Other sections get standard summaries
                summary_length = 100
            
            # Summarize the section
            if len(section_text.split()) > 200:
                # Use the main summarizer for longer sections
                summary = self.summarizer(
                    section_text,
                    max_length=summary_length,
                    min_length=summary_length // 2,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    truncation=True
                )
                
                if summary and len(summary) > 0:
                    return summary[0]['summary_text']
            else:
                # For shorter sections, use extractive summarization
                return self._extractive_summarize(section_text, summary_length)
            
            return ""
            
        except Exception as e:
            print(f"Error summarizing section {section_index}: {e}")
            return ""
    
    def _summarize_with_chunks(self, text: str, max_length: int, min_length: int) -> str:
        """
        Summarize long text by breaking it into chunks and combining summaries
        """
        try:
            # Split text into chunks of manageable size
            chunk_size = 800  # words per chunk
            words = text.split()
            chunks = []
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                if len(chunk.strip()) > 100:
                    chunks.append(chunk)
            
            if len(chunks) == 1:
                # Single chunk, summarize directly
                return self._summarize_single_chunk(chunks[0], max_length, min_length)
            
            # Summarize each chunk
            chunk_summaries = []
            for chunk in chunks:
                try:
                    chunk_summary = self._summarize_single_chunk(chunk, max_length // len(chunks), min_length // len(chunks))
                    if chunk_summary:
                        chunk_summaries.append(chunk_summary)
                except Exception as e:
                    print(f"Error summarizing chunk: {e}")
                    continue
            
            if not chunk_summaries:
                return self._fallback_summarize(text)
            
            # Combine chunk summaries
            combined_summary = ' '.join(chunk_summaries)
            
            # If combined summary is too long, summarize it again
            if len(combined_summary.split()) > max_length:
                final_summary = self.summarizer(
                    combined_summary,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    truncation=True
                )
                
                if final_summary and len(final_summary) > 0:
                    return final_summary[0]['summary_text']
            
            return combined_summary
            
        except Exception as e:
            print(f"Error in chunk-based summarization: {e}")
            return self._fallback_summarize(text)
    
    def _summarize_single_chunk(self, chunk: str, max_length: int, min_length: int) -> str:
        """
        Summarize a single chunk of text
        """
        try:
            if not self.summarizer:
                return self._extractive_summarize(chunk, max_length)
            
            summary = self.summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                truncation=True
            )
            
            if summary and len(summary) > 0:
                return summary[0]['summary_text']
            else:
                return self._extractive_summarize(chunk, max_length)
                
        except Exception as e:
            print(f"Error summarizing single chunk: {e}")
            return self._extractive_summarize(chunk, max_length)
    
    def _extractive_summarize(self, text: str, max_length: int) -> str:
        """
        Extract-based summarization for when neural summarization fails
        """
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) <= 3:
                return text
            
            # Score sentences based on word frequency and position
            word_freq = {}
            for sentence in sentences:
                words = sentence.lower().split()
                for word in words:
                    if len(word) > 3:
                        word_freq[word] = word_freq.get(word, 0) + 1
            
            # Score each sentence
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                # Base score from word frequency
                base_score = sum(word_freq.get(word.lower(), 0) for word in sentence.split() if len(word) > 3)
                
                # Bonus for first and last sentences (introduction and conclusion)
                if i == 0 or i == len(sentences) - 1:
                    base_score *= 1.5
                
                sentence_scores[sentence] = base_score
            
            # Get top sentences
            target_sentences = max(3, max_length // 50)  # Approximate words per sentence
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:target_sentences]
            
            # Sort by original order
            top_sentences.sort(key=lambda x: sentences.index(x[0]))
            
            summary = ' '.join([sentence for sentence, score in top_sentences])
            
            # Ensure summary is not too long
            if len(summary) > max_length * 2:
                summary = summary[:max_length * 2] + '...'
            
            return summary
            
        except Exception as e:
            print(f"Error in extractive summarization: {e}")
            return text[:max_length] + '...' if len(text) > max_length else text
    
    def _enhance_summary(self, summary: str, original_text: str) -> str:
        """
        Enhance the generated summary for better quality and readability
        """
        try:
            # Ensure proper sentence structure
            if not summary.endswith(('.', '!', '?')):
                summary += '.'
            
            # Add context if summary seems too generic
            if len(summary.split()) < 50:
                # Extract key terms from original text
                key_terms = self._extract_key_terms(original_text)
                if key_terms:
                    summary += f" Key concepts include: {', '.join(key_terms[:3])}."
            
            # Ensure the summary starts with a capital letter
            if summary and summary[0].islower():
                summary = summary[0].upper() + summary[1:]
            
            return summary
            
        except Exception as e:
            print(f"Error enhancing summary: {e}")
            return summary
    
    def _extract_key_terms(self, text: str) -> list:
        """
        Extract key technical terms from the text
        """
        try:
            # Simple key term extraction based on capitalization and frequency
            words = text.split()
            key_terms = []
            
            for word in words:
                # Look for capitalized words that might be technical terms
                if (word[0].isupper() and len(word) > 3 and 
                    word.lower() not in ['the', 'this', 'that', 'with', 'from', 'into', 'during', 'including', 'until', 'against', 'among', 'throughout', 'despite', 'towards', 'upon']):
                    key_terms.append(word)
            
            # Remove duplicates and limit results
            key_terms = list(set(key_terms))[:5]
            return key_terms
            
        except Exception as e:
            print(f"Error extracting key terms: {e}")
            return []
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess the input text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        
        # Remove LaTeX-like commands
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Remove citations
        text = re.sub(r'\[[^\]]*\]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text
    
    def _fallback_summarize(self, text: str) -> str:
        """
        Fallback summarization method using extractive approach with more detailed output
        """
        try:
            if not text or len(text.strip()) < 20:
                return "Text too short for summarization."
            
            # Simple extractive summarization
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) <= 3:
                return text
            
            # Score sentences based on word frequency
            word_freq = {}
            for sentence in sentences:
                words = sentence.lower().split()
                for word in words:
                    if len(word) > 3:  # Only consider words longer than 3 characters
                        word_freq[word] = word_freq.get(word, 0) + 1
            
            # Score each sentence
            sentence_scores = {}
            for sentence in sentences:
                score = sum(word_freq.get(word.lower(), 0) for word in sentence.split() if len(word) > 3)
                sentence_scores[sentence] = score
            
            # Get top 5 sentences instead of 3 for more detailed summary
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Sort by original order
            top_sentences.sort(key=lambda x: sentences.index(x[0]))
            
            summary = ' '.join([sentence for sentence, score in top_sentences])
            
            # Ensure summary is not too long but allow for more detail
            if len(summary) > 500:  # Increased from 300 for more detailed summaries
                summary = summary[:500] + '...'
            
            return summary
            
        except Exception as e:
            print(f"Fallback summarization failed: {e}")
            # Return first few sentences as last resort
            try:
                sentences = text.split('.')
                # Take more sentences for more detailed fallback
                summary = '. '.join(sentences[:5]) + '.'  # Increased from 3 to 5
                if len(summary) > 500:  # Increased from 300
                    summary = summary[:500] + '...'
                return summary
            except:
                return f"Summarization failed. Original text: {text[:300]}{'...' if len(text) > 300 else ''}"  # Increased from 200 to 300
    
    def summarize_with_keywords(self, text: str, keywords: list, max_length: int = 300) -> str:
        """
        Summarize text with focus on specific keywords
        """
        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Highlight keywords in text
            highlighted_text = self._highlight_keywords(cleaned_text, keywords)
            
            # Generate summary
            summary = self.summarize(highlighted_text, max_length)
            
            return summary
            
        except Exception as e:
            print(f"Error in keyword-based summarization: {e}")
            return self.summarize(text, max_length)
    
    def _highlight_keywords(self, text: str, keywords: list) -> str:
        """
        Highlight keywords in text to improve summarization focus
        """
        try:
            highlighted_text = text
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    # Add emphasis around keywords
                    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                    highlighted_text = pattern.sub(f"**{keyword}**", highlighted_text)
            
            return highlighted_text
            
        except Exception as e:
            print(f"Error highlighting keywords: {e}")
            return text
    
    def get_summary_statistics(self, text: str, summary: str) -> dict:
        """
        Get statistics about the summarization
        """
        try:
            original_words = len(text.split())
            summary_words = len(summary.split())
            compression_ratio = summary_words / original_words if original_words > 0 else 0
            
            return {
                'original_length': original_words,
                'summary_length': summary_words,
                'compression_ratio': round(compression_ratio, 3),
                'reduction_percentage': round((1 - compression_ratio) * 100, 1)
            }
            
        except Exception as e:
            print(f"Error calculating summary statistics: {e}")
            return {}
    
    def batch_summarize(self, texts: list, max_length: int = 300) -> list:
        """
        Summarize multiple texts in batch
        """
        summaries = []
        
        for text in texts:
            try:
                summary = self.summarize(text, max_length)
                summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing text: {e}")
                summaries.append("Error in summarization")
        
        return summaries
