# Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/JackRobertson77/MSU_Curriculum_Maps.git
cd MSU_Curriculum_Maps
```

## 2. Install Dependencies

```python
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt

```

## 3. Add Example Data

Place the example data files in the project root (same folder as cleaning.py):

Fake_registrar.csv

Fake_majors.csv

These are synthetic test datasets that match the expected column structure of the real files.

## 4. Run the Example
   
```python
python3 cleaning.py \
  --registrar Fake_registrar.csv \
  --majors Fake_majors.csv

```

Expected Output

Both CSVs load successfully.

Prerequisite relationships and course mappings are parsed.

A cleaned/merged output is produced (per your script’s logic).

## Recommended Project Structure
project-root/
├── cleaning.py
├── Fake_registrar.csv
├── Fake_majors.csv
├── requirements.txt
└── README.md
