# MSU Curriculum Mapping and Dependency Visualization  
*Data Cleaning and Graph Construction*  
*Jack Robertson, Lauryn Crandall, Livia Perelli, James Restaneo*

---
## Install Instructions
Go to install.md for intructions on how to run and install the program.

---

## Project Overview  
This project develops a robust and reusable system for **parsing, cleaning, and visualizing course dependency data** at Michigan State University. The work focuses on transforming messy course prerequisite/co-requisite data into a **directed graph**, enabling analysis of curriculum structure and student pathways.  

---

## Goals  
- Clean and standardize course requirement data.  
- Parse and interpret prerequisite logic.  
- Construct a directed graph of course/program dependencies.  
- Create static and interactive **visualizations** of the curriculum.  
- Explore relationships between curriculum structure and course outcomes (grades, DFW rates, modality).  

---

## Research Questions  
- How can we reliably extract structured prerequisite relationships from messy data?  
- What does the curriculum structure look like as a dependency graph?  
- Which courses serve as bottlenecks or critical paths?  
- How do student outcomes align with curriculum structure?  

---

## Data  
- **Primary source:** University academic programs and course requirements dataset.  
- **Supplementary source:** [MSU Course Catalog](https://reg.msu.edu/courses/search.aspx) for prerequisites and co-requisites.  

---

## Tools & Technologies  
- **Language:** Python  
- **Libraries:** `pandas`, `networkx`, `matplotlib`, `plotly`, `re` (regular expressions)  
- **Workflow:** Jupyter Notebooks  

---

## Deliverables  
- Cleaned & structured dataset  
- Python scripts for parsing and graph construction  
- Visualizations of curriculum structure (static + interactive)  
- Final report summarizing findings  
- *(Stretch Goal)* Interactive exploration tool for the curriculum graph  

---

## Risks & Backup Plan  
- **Challenge:** Input data may be inconsistent or incomplete.  
- **Fallback:** Develop **semi-automated tools** to support manual cleanup.  
- Note: The project is a **prototype/analysis pipeline**, not production-ready software.  

---

## Community Partner  
- **Sponsor:** Michigan State University academic unit
  
---

## Project Videos
Project Plan Video: https://mediaspace.msu.edu/media/MSU_Curriculum-CMSE495_Plan_Presentation_Video/1_ugrk2r6p

Final Video: https://mediaspace.msu.edu/media/MSU+Curriculum+Maps+Final+Video/1_vzu7h80f

---
## Figure Reproducibility
To reproduce the figures used in this project, open and run:
- [`reproducibility.ipynb`](./reproducibility.ipynb)

This notebook provides complete step-by-step instructions for generating the curriculum/prerequisite graph figure.
---

## License  
This project is licensed under the [MIT License](LICENSE.txt).  
