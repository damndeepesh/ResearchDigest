from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import sqlite3
import json
import os
from datetime import datetime, timedelta
from research_fetcher import ResearchFetcher
from paper_summarizer import PaperSummarizer
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize components
research_fetcher = ResearchFetcher()
paper_summarizer = PaperSummarizer()

# Database initialization
def init_db():
    conn = sqlite3.connect('research_digest.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            authors TEXT,
            abstract TEXT,
            summary TEXT,
            arxiv_id TEXT,
            published_date TEXT,
            keywords TEXT,
            text_length INTEGER DEFAULT 0,
            summary_source TEXT DEFAULT 'abstract',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def migrate_db():
    """Add missing columns to existing database if they don't exist"""
    conn = sqlite3.connect('research_digest.db')
    cursor = conn.cursor()
    
    try:
        # Check if text_length column exists
        cursor.execute("PRAGMA table_info(papers)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add text_length column if it doesn't exist
        if 'text_length' not in columns:
            cursor.execute('ALTER TABLE papers ADD COLUMN text_length INTEGER DEFAULT 0')
            print("Added text_length column to papers table")
        
        # Add summary_source column if it doesn't exist
        if 'summary_source' not in columns:
            cursor.execute('ALTER TABLE papers ADD COLUMN summary_source TEXT DEFAULT "abstract"')
            print("Added summary_source column to papers table")
        
        # Update existing records to have default values for new columns
        cursor.execute('UPDATE papers SET text_length = 0 WHERE text_length IS NULL')
        cursor.execute('UPDATE papers SET summary_source = "abstract" WHERE summary_source IS NULL')
        
        conn.commit()
        print("Database migration completed successfully")
        
    except Exception as e:
        print(f"Error during database migration: {e}")
        conn.rollback()
    finally:
        conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to Research Digest'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('search_papers')
def handle_search_papers(data):
    try:
        query = data.get('query', '')
        print(f"Searching for papers with query: {query}")
        
        # Emit status update
        emit('status', {'message': f'Searching for papers on: {query}'})
        
        # Fetch papers from arXiv
        papers = research_fetcher.search_papers(query, max_results=5)
        
        print(f"Received {len(papers) if papers else 0} papers from research fetcher")
        if papers:
            print(f"First paper structure: {papers[0]}")
        
        if papers:
            # Store papers in session for later reference
            session['current_papers'] = papers
            
            # Check which papers already have summaries in the database
            conn = sqlite3.connect('research_digest.db')
            cursor = conn.cursor()
            
            # Get existing summaries for these papers
            existing_summaries = {}
            for paper in papers:
                arxiv_id = paper.get('arxiv_id', '')
                if arxiv_id:
                    cursor.execute('SELECT summary FROM papers WHERE arxiv_id = ?', (arxiv_id,))
                    result = cursor.fetchone()
                    if result and result[0]:
                        existing_summaries[arxiv_id] = result[0]
            
            conn.close()
            
            # Format papers for display
            formatted_papers = []
            for i, paper in enumerate(papers):
                try:
                    # Validate paper structure and provide defaults for missing fields
                    title = paper.get('title', 'Title not available')
                    authors = paper.get('authors', ['Unknown Author'])
                    abstract = paper.get('abstract', 'Abstract not available')
                    arxiv_id = paper.get('arxiv_id', f'paper_{i}')
                    published_date = paper.get('published_date', 'Date not available')
                    
                    # Ensure authors is a list
                    if isinstance(authors, str):
                        authors = [authors]
                    
                    # Truncate abstract if too long
                    if len(abstract) > 200:
                        abstract = abstract[:200] + '...'
                    
                    # Skip papers with no title or abstract
                    if title == 'Title not available' and abstract == 'Abstract not available':
                        print(f"Skipping paper {i} due to missing title and abstract")
                        continue
                    
                    # Check if this paper already has a summary
                    has_summary = arxiv_id in existing_summaries
                    
                    formatted_papers.append({
                        'id': i,
                        'title': title,
                        'authors': authors,
                        'abstract': abstract,
                        'arxiv_id': arxiv_id,
                        'published_date': published_date,
                        'has_summary': has_summary
                    })
                    
                    print(f"Formatted paper {i}: {title} (has_summary: {has_summary})")
                    
                except Exception as paper_error:
                    print(f"Error formatting paper {i}: {paper_error}")
                    print(f"Paper data: {paper}")
                    continue
            
            if formatted_papers:
                emit('papers_found', {
                    'papers': formatted_papers,
                    'query': query
                })
            else:
                emit('error', {'message': 'No valid papers found for your query. Please try different keywords.'})
        else:
            emit('error', {'message': 'No papers found for your query. Try different keywords.'})
            
    except Exception as e:
        print(f"Error searching papers: {e}")
        emit('error', {'message': f'Error searching papers: {str(e)}'})

@socketio.on('summarize_paper')
def handle_summarize_paper(data):
    try:
        paper_index = data.get('paper_index', 0)
        papers = session.get('current_papers', [])
        
        if not papers or paper_index >= len(papers):
            emit('error', {'message': 'Paper not found. Please search again.'})
            return
        
        selected_paper = papers[paper_index]
        
        # Validate paper structure
        title = selected_paper.get('title', 'Title not available')
        authors = selected_paper.get('authors', ['Unknown Author'])
        abstract = selected_paper.get('abstract', 'Abstract not available')
        arxiv_id = selected_paper.get('arxiv_id', f'paper_{paper_index}')
        published_date = selected_paper.get('published_date', 'Date not available')
        
        # Ensure authors is a list
        if isinstance(authors, str):
            authors = [authors]
        
        # Check if paper already exists in database and has a summary
        conn = sqlite3.connect('research_digest.db')
        cursor = conn.cursor()
        cursor.execute('SELECT summary FROM papers WHERE arxiv_id = ?', (arxiv_id,))
        existing = cursor.fetchone()
        
        if existing and existing[0]:
            # Paper already has a summary, send it directly
            summary = existing[0]
            emit('status', {'message': 'Using existing summary'})
            
            # Get additional information from database
            cursor.execute('SELECT text_length, summary_source FROM papers WHERE arxiv_id = ?', (arxiv_id,))
            additional_info = cursor.fetchone()
            text_length = additional_info[0] if additional_info else 0
            summary_source = additional_info[1] if additional_info else 'abstract'
            
            # Send existing summary to client
            emit('paper_summary', {
                'title': title,
                'authors': authors,
                'summary': summary,
                'arxiv_id': arxiv_id,
                'full_abstract': abstract,
                'already_summarized': True,  # Flag to indicate this was already summarized
                'summary_source': summary_source,
                'text_length': text_length
            })
            
            conn.close()
            return
        
        # Paper doesn't have a summary, generate new one from full paper
        emit('status', {'message': f'Downloading full paper: {title}'})
        
        # Get full paper text
        full_paper = research_fetcher.get_paper_with_full_text(arxiv_id)
        
        if not full_paper or not full_paper.get('full_text'):
            emit('status', {'message': 'Failed to download full paper, using abstract instead'})
            # Fallback to abstract summarization
            if not abstract or abstract == 'Abstract not available' or len(abstract.strip()) < 50:
                emit('error', {'message': 'Cannot generate summary: Abstract is too short or unavailable.'})
                return
            
            summary = paper_summarizer.summarize(abstract)
        else:
            emit('status', {'message': 'Processing full paper content...'})
            
            # Use full paper text for summarization
            full_text = full_paper['full_text']
            text_length = full_paper.get('text_length', 0)
            
            print(f"Processing full paper with {text_length} characters")
            
            # Generate comprehensive summary from full paper
            summary = paper_summarizer.summarize(full_text, max_length=800, min_length=400)
            
            emit('status', {'message': f'Generated comprehensive summary from {text_length} characters of text'})
        
        # Store in database
        cursor.execute('''
            INSERT OR REPLACE INTO papers 
            (title, authors, abstract, summary, arxiv_id, published_date, keywords, text_length, summary_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            title,
            ', '.join(authors),
            abstract,
            summary,
            arxiv_id,
            published_date,
            selected_paper.get('keywords', ''),
            full_paper.get('text_length', 0) if full_paper else 0,
            'full_paper' if full_paper and full_paper.get('full_text') else 'abstract'
        ))
        conn.commit()
        conn.close()
        
        # Send new summary to client
        emit('paper_summary', {
            'title': title,
            'authors': authors,
            'summary': summary,
            'arxiv_id': arxiv_id,
            'full_abstract': abstract,
            'already_summarized': False,  # Flag to indicate this is a new summary
            'summary_source': 'full_paper' if full_paper and full_paper.get('full_text') else 'abstract',
            'text_length': full_paper.get('text_length', 0) if full_paper else 0
        })
        
    except Exception as e:
        print(f"Error summarizing paper: {e}")
        emit('error', {'message': f'Error summarizing paper: {str(e)}'})

@socketio.on('get_recent_papers')
def handle_get_recent_papers():
    try:
        conn = sqlite3.connect('research_digest.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT title, authors, summary, arxiv_id, published_date, created_at, text_length, summary_source
            FROM papers 
            ORDER BY created_at DESC 
            LIMIT 10
        ''')
        
        recent_papers = []
        for row in cursor.fetchall():
            recent_papers.append({
                'title': row[0],
                'authors': row[1],
                'summary': row[2],
                'arxiv_id': row[3],
                'published_date': row[4],
                'created_at': row[5],
                'text_length': row[6],
                'summary_source': row[7]
            })
        
        conn.close()
        
        emit('recent_papers', {'papers': recent_papers})
        
    except Exception as e:
        print(f"Error fetching recent papers: {e}")
        emit('error', {'message': f'Error fetching recent papers: {str(e)}'})

if __name__ == '__main__':
    init_db()
    migrate_db() # Call the new migration function
    print("Research Digest System Started!")
    print("Database initialized successfully!")
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
