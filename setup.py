from setuptools import setup

from os import path

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="polymarket-sentiment",
    version="0.0.1",
    description="Real-time sentiment analysis for Polymarket trading signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        'Homepage': 'https://github.com/joel-saucedo/ntscraper',
        'Source': 'https://github.com/joel-saucedo/ntscraper',
        'Documentation': 'https://github.com/joel-saucedo/ntscraper'
    },
    keywords=["twitter", "nitter", "scraping", "sentiment", "polymarket", "trading"],
    author="Joel Saucedo",
    author_email="your-email@example.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent"
    ],
    packages=["polymkt_sent"],
    include_package_data=True,
    install_requires=[
        "requests>=2.28", 
        "beautifulsoup4>=4.11", 
        "lxml>=4.9", 
        "tqdm>=4.66",
        "pyyaml>=6.0",
        "duckdb>=0.9.0",
        "pandas>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "vaderSentiment>=3.3.2",
        "aiohttp>=3.8.0",
        "asyncio-throttle>=1.0.0",
        "python-dateutil>=2.8.0",
        "pydantic>=2.0.0",
        "click>=8.0.0",
        "snscrape>=0.7.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "vcrpy>=4.3.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "polymkt-sent=polymkt_sent.cli:main",
        ],
    },
)