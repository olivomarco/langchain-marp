# MARP presentations with Langchain

This repo showcases how to leverage a Langchain agent with [MARP](https://marp.app/) to create presentations out of a web research.

To run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py -q "Azure AKS networking best practices"
```

To then create a PPTX/PDF presentation out of the output provided, you can use [this tool](https://github.com/marp-team/marp-cli) and run:

```bash
npx @marp-team/marp-cli@latest outputs/slide-deck.md -o outputs/output.pdf --allow-local-files
npx @marp-team/marp-cli@latest outputs/slide-deck.md -o outputs/output.pptx --allow-local-files
```
