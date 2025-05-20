from flask import Flask, request, render_template, jsonify
from flask_cors import CORS  # ✅ Import CORS
import os
import re
import hashlib
import numpy as np
import tempfile
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # ✅ Enable CORS for all routes

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()  # Use temp directory for uploads

model = SentenceTransformer('all-MiniLM-L6-v2')

def anonymize_text(text):
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    words = text.split()
    for i, word in enumerate(words):
        if i > 0 and word[0].isupper() and not words[i-1].endswith('.'):
            hash_obj = hashlib.md5(word.encode())
            words[i] = f'[ENTITY-{hash_obj.hexdigest()[:6]}]'
    return ' '.join(words)

def read_pdf(file_stream):
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_text_file(file_stream):
    content = file_stream.read()
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='replace')
    return content

def read_file_content(file):
    file_stream = io.BytesIO(file.read())
    if file.filename.lower().endswith('.pdf'):
        return read_pdf(file_stream)
    else:
        file_stream.seek(0)
        return read_text_file(file_stream)

def chunk_text(text, chunk_size=512):
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def get_embeddings(text):
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)
    return embeddings

def calculate_content_coverage(article_embeddings, synopsis_embeddings):
    max_similarities = []
    for syn_emb in synopsis_embeddings:
        similarities = cosine_similarity([syn_emb], article_embeddings)[0]
        max_similarities.append(np.max(similarities))
    avg_max_similarity = np.mean(max_similarities) if max_similarities else 0
    content_score = min(40, int(avg_max_similarity * 40))
    return content_score, max_similarities

def calculate_clarity_score(synopsis_text):
    sentences = synopsis_text.split('.')
    avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
    length_penalty = 0
    if avg_sentence_length < 5 or avg_sentence_length > 30:
        length_penalty = 5
    transition_words = ['however', 'therefore', 'consequently', 'thus', 'furthermore', 
                         'moreover', 'in addition', 'similarly', 'in contrast']
    transition_count = sum(1 for word in transition_words if word in synopsis_text.lower())
    clarity_score = min(30, 25 - length_penalty + min(5, transition_count))
    return clarity_score

def calculate_coherence_score(synopsis_embeddings):
    if len(synopsis_embeddings) <= 1:
        return 30
    coherence_scores = []
    for i in range(len(synopsis_embeddings) - 1):
        sim = cosine_similarity([synopsis_embeddings[i]], [synopsis_embeddings[i+1]])[0][0]
        coherence_scores.append(sim)
    avg_coherence = np.mean(coherence_scores) if coherence_scores else 1.0
    coherence_score = min(30, int(avg_coherence * 30))
    return coherence_score

def generate_feedback(content_score, clarity_score, coherence_score, max_similarities):
    feedback = []
    if content_score >= 35:
        feedback.append("Excellent content coverage - the synopsis captures the key points from the article effectively.")
    elif content_score >= 25:
        feedback.append("Good content coverage - most important points are included, but some details could be added.")
    else:
        feedback.append("The synopsis is missing significant content from the original article.")
    if clarity_score >= 25:
        feedback.append("The synopsis is clearly written and easy to understand.")
    elif clarity_score >= 15:
        feedback.append("The writing is somewhat clear but could benefit from improved sentence structure.")
    else:
        feedback.append("Work on clarity - consider varying sentence length and using transition words.")
    if coherence_score >= 25:
        feedback.append("The synopsis flows logically from one point to the next.")
    elif coherence_score >= 15:
        feedback.append("The synopsis could benefit from better logical flow between ideas.")
    else:
        feedback.append("Improve the coherence by ensuring points connect logically to each other.")
    if max_similarities and min(max_similarities) < 0.5:
        feedback.append("Some parts of the synopsis don't clearly relate to the original article.")
    return feedback[:3]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/score', methods=['POST'])
def score_synopsis():
    if 'article' not in request.files or 'synopsis' not in request.files:
        return jsonify({'error': 'Both article and synopsis files are required'}), 400
    article_file = request.files['article']
    synopsis_file = request.files['synopsis']
    if article_file.filename == '' or synopsis_file.filename == '':
        return jsonify({'error': 'Both files must be selected'}), 400
    try:
        article_text = read_file_content(article_file)
        synopsis_text = read_file_content(synopsis_file)
        anonymized_article = anonymize_text(article_text)
        anonymized_synopsis = anonymize_text(synopsis_text)
        article_embeddings = get_embeddings(anonymized_article)
        synopsis_embeddings = get_embeddings(anonymized_synopsis)
        content_score, max_similarities = calculate_content_coverage(article_embeddings, synopsis_embeddings)
        clarity_score = calculate_clarity_score(anonymized_synopsis)
        coherence_score = calculate_coherence_score(synopsis_embeddings)
        total_score = content_score + clarity_score + coherence_score
        feedback = generate_feedback(content_score, clarity_score, coherence_score, max_similarities)
        breakdown = {
            'Content Coverage': content_score,
            'Clarity': clarity_score,
            'Coherence': coherence_score
        }
        return jsonify({
            'score': total_score,
            'feedback': feedback,
            'breakdown': breakdown
        })
    except Exception as e:
        app.logger.error(f"Error processing files: {str(e)}")
        return jsonify({'error': 'Error processing files: ' + str(e)}), 500
    finally:
        for file in [article_file, synopsis_file]:
            if file and file.filename != '':
                try:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    app.logger.error(f"Error removing temp file: {str(e)}")

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
