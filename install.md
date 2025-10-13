# Installation & Setup



## 1. Install Python


Visit https://www.python.org/downloads/  
The website will recognize your operating system automatically.  
Click the link to "get the standalone installer for Python 3.x.x"  
Once downloaded,  
 - If you're on a windows machine:  
    - Run installer executable file.  
    - Check the box that says "Add Python3.x to PATH"  
    - Click "Install Now"  
 - If you're on a Mac Machine:  
    - Open the .pkg file and follow the prompts.  



## 2. Install Git
Visit https://git-scm.com/downloads  
Find the correct version for your operating system, and click download.  
Install Git with default settings.

## 3. Clone the Repository


In your command line terminal, run the following command:
```bash
git clone https://github.com/JackRobertson77/MSU_Curriculum_Maps.git
cd MSU_Curriculum_Maps
```
## 4. Create Virtual Environment
Open a command line terminal, and run the following command:  
```bash
python3 -m venv venv
``` 
Next, activate the virtual environment.  
 - If you're using a Windows machine:  
    - In your terminal, run the following command:
      ```bash  
      venv\Scripts\activate
      ```
 - If you're using a Mac machine:  
    - In your terminal, run the following command:
      ```bash 
      source venv/bin/activate
      ```
For either OS, you will know the Virtual Environment is active when you see (venv) in your terminal prompt. 

## 5. Install Necessary Packages
The libraries we used for this project include:  
NetworkX  
Pandas  
Open a command line terminal and run the following commands:  
```bash
pip install pandas  
pip install networkx
```

## 6. Add Example Data

Place the example data files in the project root (same folder as clean_data.py):

Fake_registrar.csv

Fake_majors.csv

These are synthetic test datasets that match the expected column structure of the real files.

## 7. Run the Example
   
```python
python clean_data.py \
  --registrar Fake_registrar.csv \
  --majors Fake_majors.csv \
  --out curriculum_table.csv

```
If you are running in Windows Powershell and the above command doesn't seem to work, try the following command with all in one line:
```python
python clean_data.py --registrar Fake_registrar.csv --majors Fake_majors.csv --out curriculum_table.csv

```

Expected Output

Both CSVs load successfully.

Prerequisite relationships and course mappings are parsed.

A cleaned/merged output is produced (per your script’s logic).

## Recommended Project Structure
```bash
project-root/
├── clean_data.py
├── Fake_registrar.csv
├── Fake_majors.csv
├── requirements.txt
└── README.md
```



# Testing the Installations

---

## 1) Verify Python is available
Run in a terminal from the project root:
```bash
python --version
```

**Pass:** command prints a Python 3.x version and exits without error.

---

## 2) Confirm example data is present
The test uses the two fake CSVs referenced in the install file.

```bash
python - << 'PY'
import os
files = ["Fake_registrar.csv", "Fake_majors.csv"]
for f in files:
    print(f"{f} exists:", os.path.exists(f))
    if os.path.exists(f):
        print(f"  size (bytes):", os.path.getsize(f))
PY
```

Or on Windows Powershell, try running:
```python
python -c "import os; files=['Fake_registrar.csv','Fake_majors.csv']; [print(f'{f} exists:',os.path.exists(f)) or (os.path.exists(f) and print('  size (bytes):',os.path.getsize(f))) for f in files]"
```

**Pass:** both files report `exists: True` and non‑zero size.

---

## 3) Run the example pipeline

```bash
# If your repo uses `clean_data.py`:
python clean_data.py --registrar Fake_registrar.csv --majors Fake_majors.csv --out curriculum_table.csv

```

**Pass:** returns to the prompt without a Python traceback.

---

## 4) Validate the output
Check that the output CSV was created, is non‑empty, and preview a few rows **without requiring pandas**.

```bash
python - << 'PY'
import os, csv, itertools
p = "curriculum_table.csv"
print(p, "exists:", os.path.exists(p))
if os.path.exists(p):
    sz = os.path.getsize(p)
    print(p, "size (bytes):", sz)
    with open(p, newline="", encoding="utf-8", errors="replace") as fh:
        r = csv.reader(fh)
        rows = list(itertools.islice(r, 10))
    print("Preview (first up to 10 rows):")
    for row in rows:
        print(row)
PY
```
Or on Windows Powershell, try running:
```python
python -c "import os,csv,itertools;p='curriculum_table.csv';print(p,'exists:',os.path.exists(p)); \
>> import sys; \
>> ( (print(p,'size (bytes):',os.path.getsize(p)), [print(r) for r in itertools.islice(csv.reader(open(p,newline='',encoding='utf-8',errors='replace')),10)]) if os.path.exists(p) else None)"
```

**Pass:** `curriculum_table.csv` exists, size > 0, and the preview prints rows.

---

## 5) Quick troubleshooting (minimal)
- Make sure you’re in the project root (the same folder as the script and fake CSVs).
- If imports fail when running the script, ensure you completed the install steps and, if applicable, run:
  ```bash
  pip install -r requirements.txt
  ```
- Re‑run steps **2–4** after fixing any issues.
