# Installation & Setup

## 1. Clone the Repository


In your command line terminal, run the following command:
```bash
git clone https://github.com/JackRobertson77/MSU_Curriculum_Maps.git
cd MSU_Curriculum_Maps
```
## 2. Create Conda Environment
Open an anaconda prompt command line terminal, and run the following command:  
```bash
conda create --prefix=./envs jupyter
```
If it asks you to proceed, type 'y' for yes.

Next, activate the environment.  
 - In your terminal, run the following command:
      ```bash  
      activate ./envs
      ```

You should see \envs at the end of your path when it is activated.

## 3. Install Necessary Packages
Use requirements.txt to install necessary packages for this program
In the terminal, run the following command:  
```bash
pip install -r requirements.txt
```

## 4. Run the Example
In the terminal, run the command: jupyter lab
```bash
jupyter lab
```
This should open up jupyter lab in your browser. From there, go into the file Graph.ipynb and follow the steps inside to complete the tutorial.


## 5 Quick troubleshooting (minimal)
- Make sure you’re in the project root (the same folder as the script and fake CSVs).
- If imports fail when running the script, ensure you completed the install steps 
- Re‑run steps **2–4** after fixing any issues.
