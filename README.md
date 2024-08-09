# Sentiment Analysis with Named Entity Recognition

This repository contains a FastAPI application that performs sentiment analysis on detected entities within a given text. The model identifies named entities (such as company names) and predicts their sentiment (positive, negative, or neutral). This project was developed as part of the "Acıkhack2024TDDİ" competition by the ARMA.AI team.

## Features

- **Named Entity Recognition (NER):** Detects entities in the text using a pre-trained NER model.
- **Sentiment Analysis:** Performs sentiment analysis on detected entities, returning the sentiment in Turkish (olumlu, olumsuz, nötr).
- **Entity Merging:** Automatically merges entities that are split (e.g., "Ziraat" and "Bankasi" into "Ziraat Bankasi") based on their proximity in the original text.

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
