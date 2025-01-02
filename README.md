
# PDF RAG Processing Toolkit

**Description**: A comprehensive toolkit for processing and categorising PDFs using advanced techniques, optimised for document intelligence workflows.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Features](#features)
5. [Contributing](#contributing)
6. [License](#license)
7. [Credits](#credits)
8. [Contact](#contact)

---

## Overview

This repository contains modules for efficiently managing, processing, and categorising PDF documents. The toolkit leverages state-of-the-art techniques such as Dense Passage Retrieval (DPR) and reranking methods to optimise workflows. 

### Key Modules
- `dpr_technique.py`: Implements the Dense Passage Retrieval (DPR) technique.
- `categorizer.py`: Provides tools for categorising PDF documents based on content.
- `pdf-rag-clean.py`: Cleans and processes PDFs for subsequent analysis.
- `helper_utils.py`: A collection of utility functions to support the main modules.
- `reranking.py`: Implements reranking algorithms to improve search result relevance.

---

## Installation

### Prerequisites
- Python 3.8 or higher.
- Recommended: Virtual environment for dependency isolation.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/KarlXMagnusson/pdf-rag-processing.git

	2.	Navigate to the project directory:

cd pdf-rag-processing


	3.	Install required dependencies:

pip install -r requirements.txt

Usage

Module Details
	1.	DPR Technique:
	•	Implements dense passage retrieval for efficient querying.
	•	Example usage:

python dpr_technique.py


	2.	Categoriser:
	•	Categorises documents based on content metadata.
	•	Example usage:

python categorizer.py


	3.	PDF Cleaning:
	•	Cleans and prepares PDF files for analysis.
	•	Example usage:

python pdf-rag-clean.py


	4.	Helper Utilities:
	•	Provides reusable functions for other modules.
	•	Not typically executed independently.
	5.	Reranking:
	•	Improves retrieval results via reranking.
	•	Example usage:

python reranking.py

Features
	•	Advanced PDF document preprocessing.
	•	Categorisation tools leveraging metadata and content.
	•	Dense Passage Retrieval for optimal query handling.
	•	Reranking to improve search result accuracy.
	•	Modular design for easy integration and customisation.

Contributing

Contributions to this project are welcome. Please follow these steps to contribute:
	1.	Fork the repository.
	2.	Create a feature branch:

git checkout -b feature-name


	3.	Commit your changes:

git commit -m "Add feature-name"


	4.	Push to your branch:

git push origin feature-name


	5.	Create a pull request for review.

Code of Conduct

By participating, you agree to abide by the Code of Conduct.

License

This project is licensed under the MIT License.

Credits

Special thanks to Paulo for their invaluable contributions to this project. Paulo’s insights and expertise were crucial in the development and refinement of the tools provided in this repository.

Contact

For further assistance or enquiries, please contact:
	•	Author: K Tage M
	•	GitHub: KarlXMagnusson
	•	Email: tage@embedded-computing.eu

Acknowledgements

This project is part of educational course by Paulo C, check his profile and community for deeper insights.

