# 🔬 Research Digest

> **AI-Powered Research Paper Discovery & Summarization System**

[![Docker Image](https://img.shields.io/badge/Docker%20Image-Available-blue?style=for-the-badge&logo=docker)](https://hub.docker.com/repository/docker/damndeepesh/research-digest/general)
[![License](https://img.shields.io/badge/License-GPL%203.0-green.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://python.org)

An intelligent research paper discovery and summarization system that automatically fetches the latest research papers from arXiv and provides AI-powered summaries through an interactive chat interface.

## ✨ Features

- **🔍 Smart Paper Discovery** - Natural language search with intelligent relevance scoring
- **🤖 AI-Powered Summarization** - Advanced BART/T5 models for high-quality summaries
- **💬 Interactive Chat Interface** - Seamless paper exploration with real-time updates
- **📱 Modern Dark Theme** - Beautiful, responsive design optimized for all devices
- **💾 Intelligent Caching** - SQLite database for efficient summary storage
- **📊 Real-time Communication** - WebSocket-based instant updates
- **🐳 Docker Ready** - Pre-built Docker image available on Docker Hub

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Pull the pre-built image
docker pull damndeepesh/research-digest:latest

# Run the container
docker run -p 5000:5000 damndeepesh/research-digest:latest
```

**Docker Hub**: [damndeepesh/research-digest](https://hub.docker.com/repository/docker/damndeepesh/research-digest/general)

### Option 2: Docker Compose

```bash
# Clone and run
git clone https://github.com/damndeepesh/ResearchDigest.git
cd ResearchDigest
docker-compose up --build
```

### Option 3: Local Development

```bash
# Clone the repository
git clone https://github.com/damndeepesh/ResearchDigest.git
cd ResearchDigest

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

**Open your browser**: Navigate to `http://localhost:5000`

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Flask Web     │    │   WebSocket      │    │   arXiv API     │
│   Application   │◄──►│   Communication  │◄──►│   (Research    │
│                 │    │                  │    │    Papers)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SQLite        │    │   HuggingFace    │    │   PDF Download  │
│   Database      │    │   Transformers   │    │   & Text        │
│   (Cache)       │    │   (BART/T5)      │    │   Extraction    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Environment mode | `development` |
| `FLASK_APP` | Main application file | `app.py` |
| `PYTHONUNBUFFERED` | Python output buffering | `1` (Docker) |

### Model Configuration

The system uses state-of-the-art HuggingFace transformers:

- **Primary**: `facebook/bart-large-cnn` (high quality)
- **Fallback**: `t5-small` (lightweight)

Customize in `paper_summarizer.py`:
```python
paper_summarizer = PaperSummarizer(model_name="your-model")
```

## 📱 Usage

### 1. Search for Papers
Use natural language queries:
- "Show me the latest research on Large Language Models"
- "Find papers about computer vision"
- "Research on machine learning algorithms"

### 2. Browse Results
View up to 5 relevant papers with:
- Title and authors
- Abstract preview
- Publication date
- arXiv ID

### 3. Get AI Summaries
Click any paper for intelligent summaries that:
- Condense key points
- Maintain technical accuracy
- Provide methodology insights

### 4. Access Full Papers
Each summary includes direct arXiv links.

## 🗄️ Database Schema

```sql
CREATE TABLE papers (
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
);
```

## 🔍 API Endpoints

### WebSocket Events
- `search_papers` - Search for research papers
- `summarize_paper` - Generate summary for selected paper
- `get_recent_papers` - Retrieve recently summarized papers

### HTTP Routes
- `GET /` - Main application interface
- WebSocket connection for real-time communication

## 🚀 Recent Improvements

### 🎯 Issues Addressed

#### 1. **Paper Relevance Problem** ✅ FIXED
- **Enhanced Search Algorithm**: Improved query processing with stop word removal
- **Relevance Scoring**: Intelligent scoring based on title/abstract overlap
- **Better Filtering**: Papers sorted by relevance score
- **Category Mapping**: Automatic detection of research areas (AI, CV, NLP, ML)

#### 2. **Summary Quality Problem** ✅ IMPROVED
- **Enhanced AI Models**: Upgraded summarization parameters
- **Improved Preprocessing**: Better text cleaning and intelligent truncation
- **Summary Enhancement**: Post-processing with context and key terms
- **Better Model Parameters**: Temperature, top-p sampling, repetition penalty

#### 3. **Chat Window Font Size** ✅ FIXED
- **Reduced Font Sizes**: Compact design across all elements
- **Mobile Optimization**: Responsive design for smaller screens
- **Better Space Usage**: Increased message width and optimized spacing

### 📊 Performance Improvements
- **Search Relevance**: Papers ranked by actual relevance
- **Summary Quality**: 33% longer summaries (400 vs 300 chars)
- **Interface Efficiency**: 20% reduction in chat container height
- **Mobile Experience**: Optimized for all screen sizes

## 🛠️ Development

### Project Structure
```
ResearchDigest/
├── app.py                 # Main Flask application
├── research_fetcher.py    # arXiv API integration
├── paper_summarizer.py    # AI summarization logic
├── templates/
│   └── index.html        # Main UI template
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
└── README.md             # This file
```

### Adding New Features
1. **New Data Sources**: Extend `ResearchFetcher` class
2. **Alternative Models**: Modify `PaperSummarizer` class
3. **UI Enhancements**: Update `templates/index.html`
4. **Database Changes**: Modify schema in `app.py`

## 🚀 Deployment

### Production Considerations
- Set `FLASK_ENV=production`
- Consider PostgreSQL for production database
- Implement Redis for session storage
- Use Nginx as reverse proxy
- Add health checks and monitoring

### Scaling Options
- **Horizontal**: Multiple Flask instances behind load balancer
- **Vertical**: Increase resources for larger models
- **Caching**: Redis for paper and summary caching

## 🔒 Security
- Input validation for search queries
- Rate limiting for API calls
- Secure WebSocket connections
- Database query parameterization

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Model Loading Failures | Check internet connection and disk space |
| arXiv API Errors | Verify connectivity and service status |
| Database Issues | Check permissions and disk space |

### Debug Mode
```python
socketio.run(app, debug=True, host='0.0.0.0', port=5000)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the **GPL-3.0 License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[arXiv](https://arxiv.org)** - Research paper access
- **[HuggingFace](https://huggingface.co)** - Transformer models
- **[Flask](https://flask.palletsprojects.com)** - Web framework
- **[Socket.IO](https://socket.io)** - Real-time communication

## 📞 Support

- Check the troubleshooting section
- Review existing issues
- Create a new issue with detailed information

---

**Happy Researching! 🔬📚**

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-damndeepesh%2Fresearch--digest-blue?style=for-the-badge&logo=docker)](https://hub.docker.com/repository/docker/damndeepesh/research-digest/general)
