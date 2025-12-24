from setuptools import setup, find_packages

setup(
    name="multimodal-math-mentor",
    version="1.0.0",
    description="Production-grade AI system for solving JEE math problems",
    author="AI Engineering Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "streamlit>=1.35.0",
        "langchain>=0.1.14",
        "langchain-groq>=0.1.3",
        "faiss-cpu>=1.7.4",
        "paddleocr>=2.7.0.3",
        "openai-whisper>=20231214",
        "numpy>=1.26.3",
        "pydantic>=2.5.0",
    ],
    entry_points={
        "console_scripts": [
            "math-mentor=ui.app:main",
        ],
    },
)
