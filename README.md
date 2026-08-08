# BioScan AI - Facial Recognition Dashboard

**Live Demo**: [https://facial-recognition-app-tr4c.onrender.com](https://facial-recognition-app-tr4c.onrender.com)

BioScan AI is a modern, real-time facial recognition web application. It combines browser-side GPU-accelerated face tracking and landmark modeling with a lightweight Flask backend and a database to store user face templates, session logs, and face crop snapshots.

---

## Key Features
- **GPU-Accelerated Scanner**: Real-time browser-based face tracking at 30 FPS using TensorFlow.js and `face-api.js`.
- **Hybrid Architecture**: Extraction of 128-dimensional face descriptors is offloaded to the client, while secure, fast Euclidean distance matching is processed on the Python backend.
- **Multiple Profile Templates**: Register multiple angles/expressions under the same user name to enhance detection accuracy.
- **Visual Log Timeline**: Saves 150x150 square face crop snapshots to the file-system and displays them in a chronological dashboard log with confidence indicators.
- **Glassmorphic UI**: High-fidelity responsive dark mode design utilizing glowing badges, pulsing scanner overlays, and modern typography.
- **Fully Dockerized**: Built with persistence volumes to preserve profiles and snapshots outside container lifetimes.

---

## Tech Stack
- **Backend**: Flask, NumPy, Pillow, Python 3.11
- **Database**: SQLite & Flask-SQLAlchemy (ORM)
- **Frontend**: Vanilla HTML5/CSS3, JavaScript (ES6+), face-api.js, FontAwesome v6
- **DevOps**: Docker, Docker Compose

---

## File Structure
```text
├── app.py                  # Flask application & HTTP/JSON API routes
├── models.py               # SQLite database schemas (SQLAlchemy)
├── setup_models.py         # Utility script to download face-api model weights
├── requirements.txt        # Python dependency file
├── Dockerfile              # Docker container configuration
├── docker-compose.yml      # Orchestration & Volume Mount specs
├── .gitignore              # Git ignored files
├── templates/
│   └── index.html          # HTML Dashboard structure
└── static/
    ├── css/
    │   └── style.css       # Premium responsive styling (Glassmorphism)
    └── js/
        └── app.js          # Webcam loop, face-api tracking, canvas drawing
```

---

## How to Run Locally

### Option A: Using Docker (Recommended)
Make sure you have Docker and Docker Compose installed, then run:
```bash
# Rebuild and start container in detached background mode
docker compose up --build -d
```
Navigate to **http://localhost:5000** in your browser.

### Option B: Using Python Virtual Environment
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model setup script to download the weight shards locally:
   ```bash
   python setup_models.py
   ```
4. Start the Flask application:
   ```bash
   python app.py
   ```
5. Open your browser to **http://127.0.0.1:5000**.

---

## Deploying to Render

Yes, this application will deploy and run perfectly on **Render**!

Because of the hybrid structure, the recommended way to host this on Render is as a **Web Service using the Docker runtime**:

### Deployment Steps on Render:
1. Push your repository to **GitHub**.
2. Log into **Render** and create a new **Web Service**.
3. Select your GitHub repository.
4. Render will automatically detect the `Dockerfile` and configure the runtime environment to **Docker**.
5. Choose the **Free** instance type (or any other tier) and click **Deploy Web Service**.
6. Render will build the image, run `setup_models.py` inside the container build environment to download the models, and deploy the application.

### Database Persistence Options on Render

Render's local file-system is **ephemeral** (gets wiped out and reset on redeployments or when the instance spins down due to inactivity on the free tier). You have two options to keep your database safe:

#### Option A: External Cloud PostgreSQL (100% Free)
You can connect a free PostgreSQL database (such as **Render PostgreSQL**, **Neon.tech**, or **Supabase**) to this application.
1. Create your PostgreSQL database and copy its **Internal Connection URL**.
2. In your Render Web Service dashboard, go to **Environment** and click **Add Environment Variable**.
3. Add **`DATABASE_URL`** as the key and paste your PostgreSQL connection string as the value.
4. Save changes. The Flask app will automatically detect it, install the necessary drivers (`psycopg2-binary`), and switch from SQLite to your persistent cloud PostgreSQL database.

#### Option B: SQLite Persistent Disk (Paid, ~$5/month)
If you wish to stick with the local SQLite file-system database but make it permanent, you can attach a **Persistent Disk** in your Render dashboard under the **Disks** tab:
- **Mount Path**: `/app/instance` (persists SQLite database file)
- **Mount Path**: `/app/static/snapshots` (persists captured face snapshots)
