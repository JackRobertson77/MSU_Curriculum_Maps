# Installation & Setup



## 1. Install Python

```python
Visit https://www.python.org/downloads/
The website will recognize your operating system automatically.
Click the link to "get the standalone installer for Python 3.x.x"
Once downloaded,
   If you're on a windows machine:
      Run installer executable file.
      Check the box that says "Add Python3.x to PATH"
      Click "Install Now"
   If you're on a Mac Machine:
      Open the .pkg file and follow the prompts.


```
## 2. Install Git
Visit https://git-scm.com/downloads  
Find the correct version for your operating system, and click download.  
Install Git with default settings.


## 3. Install Necessary Packages
The libraries we used for this project include:  
NetworkX  
Pandas  
Open a command line terminal and run the following commands:  
pip install pandas  
pip install networkx

## 4. Clone the Repository

```bash
In your command line terminal, run the following command:
git clone https://github.com/JackRobertson77/MSU_Curriculum_Maps.git
cd MSU_Curriculum_Maps
```
## 5. Create Virtual Environment
Open a command line terminal, and run the following command:  
python3 -m venv venv  
Next, activate the virtual environment.  
If you're using a Windows machine:  
   In your terminal, run the following command:  
      venv\Scripts\activate
If you're using a Mac machine:  
   In your terminal, run the following command:  
      source venv/bin/activate
For either OS, you will know the Virtual Environment is active when you see (venv) in your terminal prompt. 

## 6. Add Example Data

Place the example data files in the project root (same folder as cleaning.py):

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

Expected Output

Both CSVs load successfully.

Prerequisite relationships and course mappings are parsed.

A cleaned/merged output is produced (per your script’s logic).

## Recommended Project Structure
```bash
project-root/
├── cleaning_data.py
├── Fake_registrar.csv
├── Fake_majors.csv
├── requirements.txt
└── README.md
