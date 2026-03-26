from flask import Flask, request, jsonify, render_template, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import pickle
import numpy as np
import requests


# PDF
from PyPDF2 import PdfReader

# Vector (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

# LLM
from groq import Groq

app = Flask(__name__)
app.secret_key = "secret123"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.unauthorized_handler
def unauthorized():
    from flask import request, redirect
    
    # Agar API call hai → JSON
    if request.path.startswith("/chat") or request.path.startswith("/upload"):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Normal page → login page
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
    role = db.Column(db.String(10))  # user / assistant
    message = db.Column(db.Text)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/history")
# @login_required
def history():
    chats = Chat.query.filter_by(user_id=current_user.id).all()

    data = []
    for c in chats:
        data.append({
            "role": c.role,
            "message": c.message
        })

    return jsonify(data)

# ---------------- VECTOR ----------------
vectorizer = TfidfVectorizer()
index = None
texts = []

# ---------------- LLM ----------------
client = Groq(api_key="#")

# ---------------- PDF PROCESS ----------------
def process_pdf(filepath):
    global index, texts

    reader = PdfReader(filepath)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    chunk_size = 800
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    texts = chunks

    embeddings = vectorizer.fit_transform(chunks).toarray()

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
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

        with open("vectorstore/texts.pkl", "rb") as f:
            texts = pickle.load(f)

        with open("vectorstore/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

    query_vector = vectorizer.transform([query]).toarray()
    D, I = index.search(np.array(query_vector), k=5)

    results = [texts[i] for i in I[0]]
    return "\n".join(results)

# ---------------- LLM ----------------
def ask_llm(question, context):
    # load last 6 messages of user
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.id.desc()).limit(6).all()
    chats = chats[::-1]

    messages = []

    # system prompt
    messages.append({
        "role": "system",
        "content": f"You are a helpful AI tutor. Use context if provided.\n\nContext:\n{context}"
    })

    # memory
    for c in chats:
        messages.append({
            "role": c.role,
            "content": c.message
        })

    # current question
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    answer = response.choices[0].message.content

    # save chat
    db.session.add(Chat(user_id=current_user.id, role="user", message=question))
    db.session.add(Chat(user_id=current_user.id, role="assistant", message=answer))
    db.session.commit()

    return answer
def detect_mode(question):
    q = question.lower()

    # IMAGE keywords
    image_keywords = ["draw", "image", "generate", "picture", "photo", "create image"]

    for word in image_keywords:
        if word in q:
            return "image"

    # PDF keywords
    pdf_keywords = ["pdf", "document", "file", "according to pdf"]

    for word in pdf_keywords:
        if word in q:
            return "rag"

    return "chat"



def generate_image(prompt):
    API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
    
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

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

@app.route("/login", methods=["GET", "POST"])
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

@app.route("/register", methods=["GET", "POST"])
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

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

@app.route("/upload", methods=["POST"])
@login_required
def upload():
    file = request.files["file"]

    if file:
        path = os.path.join("uploads", file.filename)
        file.save(path)
        process_pdf(path)
        return jsonify({"message": "PDF processed"})

    return jsonify({"error": "No file"})

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.json
    question = data.get("message")

    mode = detect_mode(question)

    print("MODE:", mode)  # debug

    # IMAGE
    if mode == "image":
        img = generate_image(question)
        return jsonify({"image": img})

    # RAG
    context = search(question) if mode == "rag" else ""

    answer = ask_llm(question, context)

    return jsonify({"answer": answer})

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    with app.app_context():
        db.create_all()

    app.run(debug=True)
