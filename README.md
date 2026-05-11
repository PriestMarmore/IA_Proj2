# ChurnSight — E-Commerce Churn Prediction

A machine learning project built for the IART course @ L.EIC. The idea is pretty simple: given a customer's behaviour on an e-commerce platform (how often they buy, how much they spend, whether they open emails, etc.), can we predict if they're about to stop buying? Turns out, yes.

The project has two parts:
- A **Python ML model** that trains on synthetic customer data and figures out who's likely to churn
- A **Streamlit web app** that lets you plug in a customer's info and get a churn risk prediction on the spot

---

## What's in the repo

```
src/
├── ChurnModel.py       # generates data, trains the model, saves best_model.pkl
├── app.py              # the web app
├── best_model.pkl      # saved model (generated after running ChurnModel.py)
└── customer_data.csv   # synthetic dataset (generated after running ChurnModel.py)
```

---

## Setup & How to Run

The steps are basically the same across all platforms — the only differences are small things like how you activate the virtual environment.

### 1. Clone the repo / get the files

Just make sure you have the `src/` folder with `ChurnModel.py` and `app.py` in it.

### 2. Make sure Python is installed

You need Python 3.9 or higher.

- **Linux/Mac** — you probably already have it. Run `python3 --version` to check.
- **Windows** — if you don't have it, grab it from [python.org](https://python.org). During install, tick **"Add Python to PATH"** or you'll have a bad time.

### 3. Create a virtual environment

A virtual environment keeps the project's dependencies isolated so they don't mess with anything else on your machine.

**Linux / Mac:**
```bash
cd src
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
cd src
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
cd src
python -m venv venv
venv\Scripts\Activate.ps1
```

> If PowerShell complains about execution policy, run this first:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

Once activated, you should see `(venv)` at the start of your terminal prompt. You'll need to do this every time you open a new terminal.

### 4. Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn streamlit
```

This works the same on all platforms once the venv is active.

### 5. Train the model

```bash
python3 ChurnModel.py   # Linux/Mac
python ChurnModel.py    # Windows
```

This will:
- Generate 1000 synthetic customers
- Train a Decision Tree classifier
- Print accuracy, F1 score, and other metrics
- Save `best_model.pkl` and `customer_data.csv` to the same folder

### 6. Run the web app

```bash
streamlit run app.py
```

Same command on all platforms. It'll open the app automatically in your browser at `http://localhost:8501`. If it doesn't open by itself, just paste that URL manually.

---

## Troubleshooting

**`python` not found on Linux/Mac`** — use `python3` instead, that's just how most Linux/Mac systems name it.

**`best_model.pkl not found` error in the app** — you need to run `ChurnModel.py` before `app.py`. The model file gets created by the training script.

**Plots hanging / GUI window freezing on Linux** — add these two lines near the top of `ChurnModel.py`, right after the imports:
```python
import matplotlib
matplotlib.use("Agg")
```

**PowerShell won't let you activate the venv on Windows** — see the note in step 3 above about execution policy.

---

## Dependencies

| Package | What it's for |
|---|---|
| numpy | number crunching |
| pandas | data handling |
| scikit-learn | the actual ML stuff |
| matplotlib | plots |
| seaborn | prettier plots |
| streamlit | the web app |
