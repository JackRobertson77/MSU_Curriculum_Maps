# Installation & Setup

## 0. Download Anaconda/Miniconda If Not Already Installed

Download from [anaconda.com](https://www.anaconda.com/download) if needed.

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
This should open up jupyter lab in your browser. To run a demo of our api, click into the folder titled "api." From there, go to `API_Demo.ipynb` which will take you through a quick demo of the differernt functions in our api, cell by cell. 


## 5 Quick troubleshooting (minimal)
- Make sure you’re in the project root.
- If imports fail when running the script, ensure you completed the install steps 
- Re‑run steps **2–4** after fixing any issues.
