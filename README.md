# 🔍 Domain Meta Extractor - Streamlit Web App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://domain-meta-extractor.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful web application for extracting meta titles and descriptions from domain names. Built with Streamlit, this tool processes CSV files containing domains and returns enhanced data with comprehensive meta information using intelligent fallback strategies.

## ✨ Features

- **🚀 Web Interface**: Easy-to-use Streamlit app with no command line required
- **📁 File Upload**: Simply upload your CSV file with domains
- **🔄 Multiple Fallbacks**: 3-tier extraction strategy for maximum reliability
- **⚡ Real-time Progress**: Live progress tracking and status updates
- **📊 Visual Analytics**: Charts showing extraction method breakdowns
- **📥 One-Click Download**: Instant download of processed results
- **⚙️ Configurable**: Adjustable processing parameters

## 🚀 Quick Start

### Online (Recommended)
1. Visit [**domain-meta-extractor.streamlit.app**](https://domain-meta-extractor.streamlit.app)
2. Upload your CSV file with a `domain` column
3. Click "Extract Meta Information"
4. Download your enhanced results!

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/domain-meta-extractor.git
   cd domain-meta-extractor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📋 Usage Guide

### Preparing Your CSV

Your CSV file should have a column named `domain` containing the domain names you want to process:

```csv
domain
google.com
github.com
stackoverflow.com
wikipedia.org
amazon.com
```

### Processing Options

- **Maximum domains**: Limit the number of domains to process (10-1000)
- **Concurrency level**: Adjust processing speed (1-20 concurrent requests)
- **JavaScript rendering**: Enable for dynamic sites (slower but more comprehensive)

### Output Format

The downloaded CSV includes:

| Column | Description |
|--------|-------------|
| `domain` | Original domain name |
| `meta_title` | Extracted title tag |
| `meta_description` | Extracted meta description |
| `extraction_method` | Which extraction method succeeded |
| `status_code` | HTTP response code |
| `extraction_time` | Processing time per domain |
| `error_message` | Details if extraction failed |

## 🛠️ Technical Details

### Extraction Methods

The app uses a progressive 3-tier fallback approach:

1. **HTML Extractor** - Fast parsing with lxml
2. **Meta Extractor** - OpenGraph and structured data
3. **Fallback Extractor** - Alternative methods and content analysis

### Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Async Python processing with aiohttp
- **Extractors**: Modular extraction system
- **Rate Limiting**: Built-in throttling to respect servers

### Performance

- **Speed**: 5-20 domains/second (depends on server responsiveness)
- **Success Rate**: 80-95% (varies by site complexity)
- **Concurrency**: Configurable up to 20 simultaneous requests

## 📊 Example Results

```csv
domain,meta_title,meta_description,extraction_method,status_code,extraction_time,error_message
google.com,Google,Search the world's information...,html_extractor,200,0.45,
github.com,GitHub: Let's build from here,GitHub is where over 100...,html_extractor,200,0.67,
stackoverflow.com,Stack Overflow,Where developers learn...,html_extractor,200,0.52,
```

## 🌟 Advanced Features

### Progress Tracking

- Real-time progress bar
- Live success/failure counters
- Individual domain status updates
- Processing time estimates

### Analytics Dashboard

- Success rate visualization
- Extraction method distribution
- Performance metrics
- Error breakdown analysis

### Configuration Options

```python
# Example configuration
config = {
    'performance': {
        'timeout': 30,
        'concurrency': 10,
        'max_retries': 2
    },
    'extraction': {
        'enable_js_fallback': False,
        'max_content_length': 1048576
    }
}
```

## 🛠️ Development

### Project Structure

```
domain-meta-extractor/
├── app.py                      # Main Streamlit application
├── requirements_streamlit.txt  # Web app dependencies
├── extractors/                 # Extraction modules
│   ├── html_extractor.py       # HTML parsing
│   ├── meta_extractor.py       # Meta tag extraction
│   └── fallback_extractor.py   # Fallback methods
├── utils/                      # Utility modules
│   └── domain_utils.py         # Domain processing
└── data/                       # Sample data
    └── sample_domains.csv      # Example input
```

### Local Development Setup

1. **Install development dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   pip install streamlit>=1.28.0
   ```

2. **Run with hot reload**
   ```bash
   streamlit run app.py --server.runOnSave true
   ```

3. **Test with sample data**
   ```bash
   streamlit run app.py -- --data-file data/sample_domains.csv
   ```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 🚀 Deployment

### Streamlit Cloud (Easiest)

1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://share.streamlit.io)
3. Deploy with one click

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

## 📈 Performance Tips

### For Faster Processing
- Increase concurrency level (10-20)
- Disable JavaScript rendering
- Use smaller batch sizes
- Process during off-peak hours

### For Better Success Rates
- Keep concurrency moderate (5-10)
- Enable JavaScript rendering
- Use longer timeouts
- Retry failed domains

## 🔧 Troubleshooting

### Common Issues

1. **"No domain column found"**
   - Ensure your CSV has a column named `domain`
   - Check for extra spaces in column names

2. **"Processing timed out"**
   - Reduce concurrency level
   - Increase timeout settings
   - Try smaller batches

3. **"Low success rate"**
   - Enable JavaScript rendering
   - Check if domains are accessible
   - Verify network connectivity

### Debug Mode

Run with debug logging:
```bash
streamlit run app.py --logger.level=debug
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [aiohttp](https://aiohttp.readthedocs.io/) for async HTTP requests
- [lxml](https://lxml.de/) for fast HTML parsing
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing

## 📞 Support

- 🐛 [Report Issues](https://github.com/yourusername/domain-meta-extractor/issues)
- 💡 [Feature Requests](https://github.com/yourusername/domain-meta-extractor/issues)
- 📧 Email: support@example.com

---

<div align="center">
  Made with ❤️ by [Your Name]

  [![Star](https://img.shields.io/github/stars/yourusername/domain-meta-extractor.svg?style=social&label=Star)](https://github.com/yourusername/domain-meta-extractor)
  [![Fork](https://img.shields.io/github/forks/yourusername/domain-meta-extractor.svg?style=social&label=Fork)](https://github.com/yourusername/domain-meta-extractor/fork)
</div>