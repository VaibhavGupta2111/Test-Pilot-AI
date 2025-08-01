## Risk & Test Case Generator (Testpilot-AI)

An end-to-end tool that reads your issue backlog (Excel), computes risk scores (including linked-issue propagation), and auto-generates step-by-step QA test cases with expected outcomes. Ship faster, reduce manual toil, and maintain a consistent QA style across your team.

## 🚀 Features
Risk Scoring • Base priority + status score • 0.5× propagation from linked issues • Recursion-safe via memoization

Semantic Test-Case Outlines • FAISS + Sentence-Transformers search for similar issues • GPT-2 "outline" pipeline to draft concise step outlines

Refinement with Expected Outcomes • DistilGPT-2 "refine" pipeline adds “✔ Expected:” validations • Skips refinement for low-risk issues

Interactive Streamlit UI • Upload your Excel, filter by risk, preview results • Real-time progress bar during test-case generation • Download scored issues & test cases as a single Excel

CPU-Only Support • No GPU required; optimized for multi-threaded CPU

# 🛠️ Getting Started

Prerequisites
Python 3.8 or later
Git
A terminal / shell environment

# Installation
1. Clone the repo
       git clone https://github.com/your-org/testpilot.git
       cd testpilot
2. Activate your virtual environment
      # macOS/Linux
      python -m venv venv && source venv/bin/activate

      # Windows PowerShell
      python -m venv venv; .\venv\Scripts\Activate.ps1

3. Install dependencies
      pip install -r requirements.txt

      # CPU-only PyTorch
      pip uninstall -y torch torchvision torchaudio
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 📈 Usage
Streamlit App
Launch the interactive UI:

      streamlit run app.py

1.Upload your issues.xlsx.
2.Filter by risk score or linked-issue presence.
3.Click Generate Test Cases and watch the progress bar.
4.Download the combined Scored Issues + Test Cases workbook.


# Contributing
    Fork the repo
1. Create a branch (git checkout -b feature/xyz)
2. Commit & push
3. Open a Pull Request

# License
    MIT License. See LICENSE for details.
