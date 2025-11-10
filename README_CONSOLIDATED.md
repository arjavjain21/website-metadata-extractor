# 🔍 Domain Meta Extractor - Consolidated Version

A powerful single-file web application for extracting meta titles and descriptions from domain names. Built with Streamlit, this tool processes CSV files containing domains and returns enhanced data with comprehensive meta information using intelligent fallback strategies.

## ✨ Features

- **🚀 Single File**: Everything in one Python file - no complex dependencies
- **📁 File Upload**: Simply upload your CSV file with domains
- **🔄 Multiple Fallbacks**: 3-tier extraction strategy for maximum reliability
- **⚡ Real-time Progress**: Live progress tracking and status updates
- **📊 Visual Analytics**: Charts showing extraction method breakdowns
- **📥 One-Click Download**: Instant download of processed results
- **⚙️ Configurable**: Adjustable processing parameters

## 🚀 Quick Start

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements_consolidated.txt
   ```

2. **Run the app**
   ```bash
   streamlit run domain_meta_extractor_consolidated.py
   ```

3. **Open your browser** to `http://localhost:8501`

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
- **Extractors**: Built-in extraction methods (no external files needed)
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

## 🌟 Key Improvements in Consolidated Version

- **Single File**: Everything consolidated into one Python file
- **No External Dependencies**: All extractors built into the main file
- **Simplified Setup**: Just install requirements and run
- **Cleaner Codebase**: Removed redundant files and directories
- **Better Performance**: Optimized for single-file deployment
- **Easier Maintenance**: All logic in one place

## 🔧 File Structure

```
domain-meta-extraction/
├── domain_meta_extractor_consolidated.py    # Main application (all-in-one)
├── requirements_consolidated.txt            # Dependencies
├── README_CONSOLIDATED.md                   # This file
├── README.md                                # Original README
└── .git/                                   # Git repository
```

## 🚀 Deployment

### Streamlit Cloud (Easiest)

1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://share.streamlit.io)
3. Deploy with one click

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_consolidated.txt .
RUN pip install -r requirements_consolidated.txt

COPY domain_meta_extractor_consolidated.py .
EXPOSE 8501

CMD ["streamlit", "run", "domain_meta_extractor_consolidated.py", "--server.address=0.0.0.0"]
```

## 📈 Performance Tips

### For Faster Processing
- Increase concurrency level (10-20)
- Use smaller batch sizes
- Process during off-peak hours

### For Better Success Rates
- Keep concurrency moderate (5-10)
- Use longer timeouts
- Retry failed domains

## 🔧 Troubleshooting

### Common Issues

1. **"No domain column found"**
   - Ensure your CSV has a column named `domain`
   - Check for extra spaces in column names

2. **"Processing timed out"**
   - Reduce concurrency level
   - Try smaller batches

3. **"Low success rate"**
   - Check if domains are accessible
   - Verify network connectivity

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [aiohttp](https://aiohttp.readthedocs.io/) for async HTTP requests
- [lxml](https://lxml.de/) for fast HTML parsing
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing

---

<div align="center">
  Made with ❤️ | Domain Meta Extractor v2.0 (Consolidated)
</div>