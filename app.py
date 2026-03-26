from flask import Flask, request, jsonify, render_template, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import pickle
import numpy as np
import requests

from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from groq import Groq

# ---------------- ENV ----------------
HF_KEY = os.getenv("HF_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")

# ---------------- APP ----------------
app = Flask(__name__)
app.secret_key = "secret123"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ✅ smart unauthorized handler
@login_manager.unauthorized_handler
def unauthorized():
    if request.path.startswith("/chat") or request.path.startswith("/upload"):
        return jsonify({"error": "Unauthorized"}), 401
    return redirect("/login")

# folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

# ---------------- MODELS ----------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    role = db.Column(db.String(10))
    message = db.Column(db.Text)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- VECTOR ----------------
vectorizer = TfidfVectorizer()
index = None
texts = []

# ---------------- LLM ----------------
client = Groq(api_key=GROQ_KEY)

# ---------------- PDF ----------------
def process_pdf(filepath):
    global index, texts

    reader = PdfReader(filepath)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    chunks = [full_text[i:i+800] for i in range(0, len(full_text), 800)]
    texts = chunks

    embeddings = vectorizer.fit_transform(chunks).toarray()

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, "vectorstore/index.faiss")

    with open("vectorstore/texts.pkl", "wb") as f:
        pickle.dump(texts, f)

    with open("vectorstore/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

# ---------------- SEARCH ----------------
def search(query):
    global index, texts, vectorizer

    if index is None:
        index = faiss.read_index("vectorstore/index.faiss")
        texts = pickle.load(open("vectorstore/texts.pkl", "rb"))
        vectorizer = pickle.load(open("vectorstore/vectorizer.pkl", "rb"))

    query_vector = vectorizer.transform([query]).toarray()
    D, I = index.search(np.array(query_vector), k=5)

    return "\n".join([texts[i] for i in I[0]])

# ---------------- LLM SAFE ----------------
def ask_llm(question, context):
    try:
        user_id = current_user.id if current_user.is_authenticated else None

        messages = [
            {
                "role": "system",
                "content": f"You are a helpful AI tutor.\n\nContext:\n{context}"
            },
            {"role": "user", "content": question}
        ]

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )

        answer = response.choices[0].message.content

        if user_id:
            db.session.add(Chat(user_id=user_id, role="user", message=question))
            db.session.add(Chat(user_id=user_id, role="assistant", message=answer))
            db.session.commit()

        return answer

    except Exception as e:
        print("LLM ERROR:", e)
        return "❌ AI error"

# ---------------- MODE ----------------
def detect_mode(q):
    q = q.lower()
    if any(x in q for x in ["image","draw","generate","photo"]):
        return "image"
    if any(x in q for x in ["pdf","document"]):
        return "rag"
    return "chat"

# ---------------- IMAGE ----------------
def generate_image(prompt):
    API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {HF_KEY}"}

    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

    if response.status_code == 200:
        filename = "static/generated.png"
        with open(filename, "wb") as f:
            f.write(response.content)
        return "/static/generated.png"

    return None

# ---------------- ROUTES ----------------
@app.route("/")
@login_required
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(
            email=request.form["email"],
            password=request.form["password"]
        ).first()

        if user:
            login_user(user)
            return redirect("/")

    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        user = User(
            email=request.form["email"],
            password=request.form["password"]
        )
        db.session.add(user)
        db.session.commit()
        return redirect("/login")

    return render_template("register.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        question = data.get("message")

        mode = detect_mode(question)

        if mode == "image":
            return jsonify({"image": generate_image(question)})

        context = search(question) if mode == "rag" else ""
        answer = ask_llm(question, context)

        return jsonify({"answer": answer})

    except Exception as e:
        print("CHAT ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(host="0.0.0.0", port=10000, debug=True)
