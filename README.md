# MARP presentations with Langchain

This repo showcases how to leverage a Langchain agent with [MARP](https://marp.app/) to create presentations out of a research using [Brave Search API](https://brave.com/search/api/).

To create a PPTX/PDF presentation out of the output provided, you can use [this tool](https://github.com/marp-team/marp-cli) and run:

```bash
npx @marp-team/marp-cli@latest outputs/slide-deck.md -o outputs/output.pdf
npx @marp-team/marp-cli@latest outputs/slide-deck.md -o outputs/output.pptx
```
# langchain-marp
