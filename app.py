import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# Load your Q&A dataset
df = pd.read_csv("qa_data (1).csv")  # expects columns: question, answer

# Build context text from CSV
context_text = ""
for _, row in df.iterrows():
    context_text += f"Q: {row['question']}\nA: {row['answer']}\n\n"

def ask_gemini(query: str) -> str:
    """
    Ask Gemini a question using your Q&A dataset as context.
    """
    prompt = (
        f"Here is my knowledge base:\n{context_text}\n\n"
        f"User question: {query}\n"
        f"Answer based only on the knowledge base above. "
        f"If the answer is not found, provide the best possible explanation."
    )
    response = model.generate_content(prompt)
    return response.text.strip()

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    if request.method == "POST":
        user_query = request.form.get("query")
        answer = ask_gemini(user_query)
    return render_template("index.html", answer=answer)

@app.route("/chat", methods=["POST"])
def chat_api():
    user_query = request.json.get("query")
    answer = ask_gemini(user_query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    import sys
    # If you run `python app.py cli` → starts CLI chatbot
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        print("RAG Custom Q&A ChatBot")
        print("Type 'exit', 'quit', 'terminate', or 'come outside' to stop.\n")
        while True:
            try:
                user_query = input("You: ").strip()
                if user_query.lower() in ["exit", "quit", "terminate", "come outside"]:
                    print("Chatbot terminated.")
                    break
                if not user_query:
                    continue
                answer = ask_gemini(user_query)
                print(f"Bot: {answer}\n")
            except KeyboardInterrupt:
                print("\nChatbot terminated by user (Ctrl+C).")
                break
    else:
        # Default → run Flask web app
        app.run(host="0.0.0.0", port=8080)
