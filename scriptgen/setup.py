"""Setup configuration for scriptgen package."""
from setuptools import setup, find_packages

setup(
    name="scriptgen",
    version="0.2.0",
    description="Multi-agent research and content generation pipeline",
    author="Chinmay Bakhale",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.2.0",
        "langchain-google-genai>=2.0.0",
        "langchain-tavily>=0.2.0",
        "python-dotenv>=1.0.0",
        "matplotlib>=3.8.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "pytest-asyncio>=0.23.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "isort>=5.13.0",
        ]
    },
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "scriptgen=scriptgen.__main__:main",
        ],
    },
)
