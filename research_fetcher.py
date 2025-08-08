import arxiv
import requests
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any
import xml.etree.ElementTree as ET
import PyPDF2
import io
import re

class ResearchFetcher:
    def __init__(self):
        self.base_url = "https://export.arxiv.org/api/query"
        self.search_url = "https://export.arxiv.org/api/query"
        # Create a session with proper redirect handling
        self.session = requests.Session()
        self.session.max_redirects = 5
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ResearchDigest/1.0)'
        })
        
    def search_papers(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for research papers using arXiv API with improved relevance
        """
        try:
            # Clean and improve the query for better relevance
            improved_query = self._improve_search_query(query)
            
            # Use arxiv library for better results
            search = arxiv.Search(
                query=improved_query,
                max_results=max_results * 2,  # Get more results to filter for relevance
                sort_by=arxiv.SortCriterion.Relevance,  # Changed from SubmittedDate to Relevance
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in search.results():
                # Calculate relevance score based on query and content
                relevance_score = self._calculate_relevance_score(result.title, result.summary, query)
                
                paper = {
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary,
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'published_date': result.published.strftime('%Y-%m-%d'),
                    'pdf_url': result.pdf_url,
                    'keywords': self._extract_keywords(result.summary, query),
                    'relevance_score': relevance_score
                }
                papers.append(paper)
                
                if len(papers) >= max_results * 2:
                    break
            
            # Sort by relevance score and take top results
            papers.sort(key=lambda x: x['relevance_score'], reverse=True)
            papers = papers[:max_results]
            
            # Check if we got any papers
            if papers:
                print(f"Successfully found {len(papers)} relevant papers using arxiv library")
                return papers
            else:
                print("No papers found with arxiv library, trying fallback...")
                return self._fallback_search(query, max_results)
            
        except Exception as e:
            print(f"Error fetching papers from arXiv: {e}")
            # Fallback to direct API call
            return self._fallback_search(query, max_results)
    
    def _improve_search_query(self, query: str) -> str:
        """
        Improve search query for better relevance
        """
        # Remove common words that don't help with search
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'show', 'me', 'latest', 'research', 'paper', 'on'}
        
        # Split query and filter out stop words
        words = query.lower().split()
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add category-specific terms for better results
        if any(term in query.lower() for term in ['machine learning', 'ml', 'ai', 'artificial intelligence']):
            filtered_words.append('cs.AI')
        elif any(term in query.lower() for term in ['computer vision', 'cv', 'image']):
            filtered_words.append('cs.CV')
        elif any(term in query.lower() for term in ['nlp', 'natural language', 'language model', 'llm']):
            filtered_words.append('cs.CL')
        elif any(term in query.lower() for term in ['deep learning', 'neural network']):
            filtered_words.append('cs.LG')
        
        # Join words and add quotes for exact phrases
        improved_query = ' '.join(filtered_words)
        
        # If original query had quotes or specific terms, preserve them
        if '"' in query or "'" in query:
            improved_query = query
        
        return improved_query
    
    def _calculate_relevance_score(self, title: str, abstract: str, query: str) -> float:
        """
        Calculate relevance score between paper and search query
        """
        query_words = set(query.lower().split())
        title_words = set(title.lower().split())
        abstract_words = set(abstract.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words -= stop_words
        title_words -= stop_words
        abstract_words -= stop_words
        
        # Calculate word overlap
        title_overlap = len(query_words & title_words) / max(len(query_words), 1)
        abstract_overlap = len(query_words & abstract_words) / max(len(query_words), 1)
        
        # Title matches are more important than abstract matches
        relevance_score = (title_overlap * 0.7) + (abstract_overlap * 0.3)
        
        return relevance_score
    
    def _fallback_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Fallback method using direct arXiv API calls
        """
        try:
            # Format query for arXiv API
            formatted_query = f"all:{query}"
            
            params = {
                'search_query': formatted_query,
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            print(f"Trying fallback search with URL: {self.search_url}")
            response = self.session.get(self.search_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response using proper XML parser
            papers = []
            content = response.text
            
            print(f"Fallback response status: {response.status_code}")
            print(f"Response content length: {len(content)}")
            
            # Use proper XML parsing
            if 'entry' in content:
                papers = self._parse_arxiv_xml_proper(content, max_results)
                print(f"Parsed {len(papers)} papers from fallback search")
            else:
                print("No 'entry' tags found in response")
            
            return papers
            
        except requests.exceptions.RequestException as e:
            print(f"Request error in fallback search: {e}")
            return []
        except Exception as e:
            print(f"Fallback search failed: {e}")
            # Last resort: return mock papers
            return self._create_mock_papers(query, max_results)
    
    def _create_mock_papers(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Create mock papers as a last resort when all other methods fail
        """
        print(f"Creating {max_results} mock papers for query: {query}")
        
        mock_papers = []
        for i in range(max_results):
            paper = {
                'title': f"Research Paper on {query} - Sample {i+1}",
                'authors': [f"Researcher {i+1}"],
                'abstract': f"This is a sample abstract for research related to {query}. This paper demonstrates the capabilities of the research digest system when external APIs are unavailable.",
                'arxiv_id': f"mock_{i+1}_{hash(query) % 10000}",
                'published_date': datetime.now().strftime('%Y-%m-%d'),
                'pdf_url': f"https://arxiv.org/abs/mock_{i+1}",
                'keywords': f"{query}, research, sample, demonstration"
            }
            mock_papers.append(paper)
        
        return mock_papers
    
    def _parse_arxiv_xml_proper(self, xml_content: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Proper XML parsing for arXiv response using ElementTree
        """
        papers = []
        try:
            # Parse XML content
            root = ET.fromstring(xml_content)
            
            # Find all entry elements
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
            
            for entry in entries[:max_results]:
                paper = {}
                
                # Extract title
                title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
                if title_elem is not None and title_elem.text:
                    paper['title'] = title_elem.text.strip()
                else:
                    paper['title'] = 'Title not available'
                
                # Extract authors
                authors = []
                author_elems = entry.findall('.//{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name')
                for author_elem in author_elems:
                    if author_elem.text:
                        authors.append(author_elem.text.strip())
                
                if authors:
                    paper['authors'] = authors
                else:
                    paper['authors'] = ['Unknown Author']
                
                # Extract abstract
                summary_elem = entry.find('.//{http://www.w3.org/2005/Atom}summary')
                if summary_elem is not None and summary_elem.text:
                    paper['abstract'] = summary_elem.text.strip()
                else:
                    paper['abstract'] = 'Abstract not available'
                
                # Extract arXiv ID
                id_elem = entry.find('.//{http://www.w3.org/2005/Atom}id')
                if id_elem is not None and id_elem.text:
                    arxiv_id = id_elem.text.strip()
                    if 'arxiv.org/abs/' in arxiv_id:
                        arxiv_id = arxiv_id.split('/')[-1]
                    paper['arxiv_id'] = arxiv_id
                else:
                    paper['arxiv_id'] = f"paper_{len(papers)}"
                
                # Extract published date
                published_elem = entry.find('.//{http://www.w3.org/2005/Atom}published')
                if published_elem is not None and published_elem.text:
                    date_str = published_elem.text.strip()
                    try:
                        # Handle ISO format dates
                        if 'T' in date_str:
                            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            paper['published_date'] = date_obj.strftime('%Y-%m-%d')
                        else:
                            paper['published_date'] = date_str[:10]
                    except:
                        paper['published_date'] = date_str[:10] if len(date_str) >= 10 else 'Date not available'
                else:
                    paper['published_date'] = datetime.now().strftime('%Y-%m-%d')
                
                # Extract keywords
                paper['keywords'] = self._extract_keywords(
                    paper.get('abstract', ''), 
                    'research'
                )
                
                # Add PDF URL
                paper['pdf_url'] = f"https://arxiv.org/pdf/{paper['arxiv_id']}"
                
                papers.append(paper)
            
            print(f"Successfully parsed {len(papers)} papers from XML using proper parser")
            return papers
            
        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
            # Fallback to simple parsing
            return self._parse_arxiv_xml_simple(xml_content, max_results)
        except Exception as e:
            print(f"Error in proper XML parsing: {e}")
            # Fallback to simple parsing
            return self._parse_arxiv_xml_simple(xml_content, max_results)
    
    def _parse_arxiv_xml_simple(self, xml_content: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Simple XML parsing as fallback when proper parsing fails
        """
        papers = []
        try:
            # This is a simplified parser - in production use xml.etree.ElementTree
            lines = xml_content.split('\n')
            current_paper = {}
            
            for line in lines:
                line = line.strip()
                if '<entry>' in line:
                    current_paper = {}
                elif '<title>' in line and '</title>' in line:
                    title = line.replace('<title>', '').replace('</title>', '').strip()
                    current_paper['title'] = title
                elif '<name>' in line and '</name>' in line:
                    author = line.replace('<name>', '').replace('</name>', '').strip()
                    if 'authors' not in current_paper:
                        current_paper['authors'] = []
                    current_paper['authors'].append(author)
                elif '<summary>' in line and '</summary>' in line:
                    abstract = line.replace('<summary>', '').replace('</summary>', '').strip()
                    current_paper['abstract'] = abstract
                elif '<id>' in line and '</id>' in line:
                    arxiv_id = line.replace('<id>', '').replace('</id>', '').strip()
                    if 'arxiv.org/abs/' in arxiv_id:
                        arxiv_id = arxiv_id.split('/')[-1]
                    current_paper['arxiv_id'] = arxiv_id
                elif '<published>' in line and '</published>' in line:
                    date_str = line.replace('<published>', '').replace('</published>', '').strip()
                    try:
                        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        current_paper['published_date'] = date_obj.strftime('%Y-%m-%d')
                    except:
                        current_paper['published_date'] = date_str[:10]
                elif '</entry>' in line:
                    if current_paper and len(papers) < max_results:
                        # Add default values for missing fields
                        if 'authors' not in current_paper:
                            current_paper['authors'] = ['Unknown Author']
                        if 'abstract' not in current_paper:
                            current_paper['abstract'] = 'Abstract not available'
                        if 'arxiv_id' not in current_paper:
                            current_paper['arxiv_id'] = f"paper_{len(papers)}"
                        if 'published_date' not in current_paper:
                            if 'date_str' in locals():
                                current_paper['published_date'] = date_str[:10]
                            else:
                                current_paper['published_date'] = datetime.now().strftime('%Y-%m-%d')
                        
                        current_paper['keywords'] = self._extract_keywords(
                            current_paper.get('abstract', ''), 
                            'research'
                        )
                        
                        papers.append(current_paper.copy())
                    current_paper = {}
            
            print(f"Successfully parsed {len(papers)} papers from XML using simple parser")
            return papers
            
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return []
    
    def _extract_keywords(self, text: str, query: str) -> str:
        """
        Extract relevant keywords from text
        """
        try:
            # Simple keyword extraction based on query and text
            query_words = query.lower().split()
            text_words = text.lower().split()
            
            # Find common words (excluding common stop words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            common_words = set(query_words) & set(text_words) - stop_words
            
            # Add some technical terms that might be relevant
            tech_terms = ['model', 'learning', 'neural', 'network', 'algorithm', 'data', 'analysis', 'method']
            relevant_terms = list(common_words) + [term for term in tech_terms if term in text.lower()]
            
            return ', '.join(relevant_terms[:5])  # Limit to 5 keywords
            
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return query
    
    def get_recent_papers_by_category(self, category: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent papers from a specific category
        """
        try:
            # Map common categories to arXiv categories
            category_mapping = {
                'computer vision': 'cs.CV',
                'machine learning': 'cs.LG',
                'natural language processing': 'cs.CL',
                'large language models': 'cs.CL',
                'llms': 'cs.CL',
                'deep learning': 'cs.LG',
                'artificial intelligence': 'cs.AI',
                'ai': 'cs.AI',
                'robotics': 'cs.RO',
                'data science': 'cs.DS'
            }
            
            arxiv_category = category_mapping.get(category.lower(), 'cs.AI')
            
            search = arxiv.Search(
                query=f"cat:{arxiv_category}",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in search.results():
                paper = {
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary,
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'published_date': result.published.strftime('%Y-%m-%d'),
                    'pdf_url': result.pdf_url,
                    'keywords': self._extract_keywords(result.summary, category)
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error fetching papers by category: {e}")
            return []
    
    def get_paper_details(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific paper
        """
        try:
            # First try using arxiv library
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(search.results())
            
            paper = {
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'arxiv_id': result.entry_id.split('/')[-1],
                'published_date': result.published.strftime('%Y-%m-%d'),
                'pdf_url': result.pdf_url,
                'keywords': self._extract_keywords(result.summary, 'research')
            }
            
            return paper
            
        except Exception as e:
            print(f"Error fetching paper details with arxiv library: {e}")
            # Fallback to direct API call
            try:
                return self._get_paper_details_fallback(arxiv_id)
            except Exception as fallback_error:
                print(f"Fallback method also failed: {fallback_error}")
                return {}
    
    def _get_paper_details_fallback(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Fallback method to get paper details using direct API call
        """
        try:
            # Use direct arXiv API
            api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
            response = self.session.get(api_url, timeout=30)
            response.raise_for_status()
            
            # Parse the XML response
            root = ET.fromstring(response.text)
            
            # Find the entry element
            entry = root.find('.//{http://www.w3.org/2005/Atom}entry')
            if entry is None:
                print("No entry found in API response")
                return {}
            
            # Extract paper information
            title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else 'Title not available'
            
            authors = []
            author_elems = entry.findall('.//{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name')
            for author_elem in author_elems:
                if author_elem.text:
                    authors.append(author_elem.text.strip())
            
            if not authors:
                authors = ['Unknown Author']
            
            abstract_elem = entry.find('.//{http://www.w3.org/2005/Atom}summary')
            abstract = abstract_elem.text.strip() if abstract_elem is not None and abstract_elem.text else 'Abstract not available'
            
            published_elem = entry.find('.//{http://www.w3.org/2005/Atom}published')
            published_date = 'Date not available'
            if published_elem is not None and published_elem.text:
                try:
                    # Parse the date and format it
                    date_obj = datetime.fromisoformat(published_elem.text.replace('Z', '+00:00'))
                    published_date = date_obj.strftime('%Y-%m-%d')
                except:
                    published_date = 'Date not available'
            
            # Construct PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            
            paper = {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'arxiv_id': arxiv_id,
                'published_date': published_date,
                'pdf_url': pdf_url,
                'keywords': self._extract_keywords(abstract, 'research')
            }
            
            return paper
            
        except Exception as e:
            print(f"Error in fallback paper details: {e}")
            return {}
    
    def get_full_paper_text(self, pdf_url: str, arxiv_id: str) -> str:
        """
        Download PDF and extract full text content
        """
        try:
            print(f"Downloading PDF from: {pdf_url}")
            
            # Download the PDF using session for better redirect handling
            response = self.session.get(pdf_url, timeout=60)
            response.raise_for_status()
            
            # Extract text from PDF
            pdf_text = self._extract_text_from_pdf(response.content)
            
            if pdf_text:
                print(f"Successfully extracted {len(pdf_text)} characters from PDF")
                return pdf_text
            else:
                print("Failed to extract text from PDF")
                return ""
                
        except Exception as e:
            print(f"Error downloading/extracting PDF: {e}")
            return ""
    
    def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text content from PDF bytes
        """
        try:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            
            full_text = ""
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                except Exception as e:
                    print(f"Error extracting text from page {page_num}: {e}")
                    continue
            
            # Clean the extracted text
            cleaned_text = self._clean_pdf_text(full_text)
            
            return cleaned_text
            
        except Exception as e:
            print(f"Error in PDF text extraction: {e}")
            return ""
    
    def _clean_pdf_text(self, text: str) -> str:
        """
        Clean and preprocess extracted PDF text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\b\d+\s*of\s*\d+\b', '', text)  # "Page X of Y"
        text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)  # "Page X"
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', text)
        
        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        # Rejoin and clean up
        cleaned_text = ' '.join(cleaned_lines)
        
        # Remove multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def get_paper_with_full_text(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Get a paper with its full text content
        """
        try:
            # Get basic paper info
            paper = self.get_paper_details(arxiv_id)
            
            if paper and paper.get('pdf_url'):
                # Download and extract full text
                full_text = self.get_full_paper_text(paper['pdf_url'], arxiv_id)
                
                if full_text:
                    paper['full_text'] = full_text
                    paper['text_length'] = len(full_text)
                    print(f"Successfully extracted full text: {len(full_text)} characters")
                else:
                    print("Failed to extract full text from PDF")
                    paper['full_text'] = ""
                    paper['text_length'] = 0
            else:
                paper['full_text'] = ""
                paper['text_length'] = 0
            
            return paper
            
        except Exception as e:
            print(f"Error getting paper with full text: {e}")
            return {}
