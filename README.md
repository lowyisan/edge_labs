# 📄 Prompt for Lab Comments

Here's my code below. Could you please add inline comments to explain each section and include additional explanations for key parts along with potential questions (and answers) that might come up during my lab test? Also, add a top-level explanation on what this code is within the code itself.

# instructions for mr chat

## 🔧 Set Your API Key (Required)

pip install openai

```bash
export OPENAI_API_KEY=your_api_key_here
```

> 🔗 Source: [Best Practices for API Key Safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)

## ▶️ Example CLI Usage with GPT-4o Model

```bash
python assistant_cli.py "Hello, how are you?" --model gpt-4o
```

## ▶️ Example CLI Usage with o3-mini Model

```bash
python assistant_cli.py "Hello, how are you?" --model o3-mini-2025-01-31
```

## 🧠 Use as a Module (Non-CLI Access)

You can import and use `assistant.py` programmatically in your own Python scripts.
