#!/usr/bin/env python3
"""
MSU Curriculum Map API - FIXED VERSION
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import re
from collections import defaultdict, deque
from pathlib import Path

# ============================================================================
# DATA MODELS
# ============================================================================

class Course(BaseModel):
    course_id: str
    course_name: str
    subject: str
    course_code: str
    credits: Optional[str] = None

class PrerequisiteGroup(BaseModel):
    courses: List[str]
    operator: str = "OR"

class CoursePrerequisites(BaseModel):
    course_id: str
    prerequisites: List[PrerequisiteGroup]
    formatted: str
    depth: int

class CurriculumCourse(BaseModel):
    course_id: str
    course_name: str
    year_level: str
    term_season: str
    credits: Optional[str]
    prerequisites: str
    prerequisite_depth: int
    has_prerequisites: bool
    feasibility_warning: Optional[str]

class MajorPlan(BaseModel):
    major_code: str
    major_title: str
    plan_title: str
    courses: List[CurriculumCourse]
    total_credits: float
    
class GraphNode(BaseModel):
    id: str
    label: str
    title: str
    major: Optional[str]
    year: Optional[str]
    term: Optional[str]
    credits: Optional[str]
    depth: int

class GraphEdge(BaseModel):
    source: str
    target: str
    type: str = "prerequisite"

class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def make_course_id(subj: str, code: str) -> str:
    """Combines subject and course code into standard format"""
    subj = str(subj).strip()
    num = str(code).strip()
    num = re.sub(r"\.0$", "", num)
    return f"{subj} {num}".strip()


def extract_prereq_groups(registrar_df: pd.DataFrame) -> dict[str, list[set[str]]]:
    """Build prerequisite relationships preserving AND/OR logic"""
    prereq_groups = defaultdict(list)
    code_pat = re.compile(r"\b([A-Z]{2,4})\s+(\d{2,4}[A-Z]?)\b")

    def norm_course(token: str) -> str:
        token = str(token).strip()
        token = re.sub(r"\s+", " ", token)
        return token.upper()

    for (subj, code), group in registrar_df.groupby(["SUBJECT", "CRSE_CODE"]):
        code_clean = re.sub(r"\.0$", "", str(code)).strip()
        target = norm_course(f"{subj} {code_clean}")
        
        prereq_rows = group[
            (group.get("REQUISITE_TYPE", "").astype(str).str.upper() == "PRE") &
            (group.get("RQ_LINE_DET_TYPE", "").astype(str).str.upper() == "CLST") &
            (group["CourseList"].notna())
        ].copy()
        
        if prereq_rows.empty:
            continue
        
        and_groups = []
        
        for _, row in prereq_rows.iterrows():
            or_group = set()
            course_list = str(row.get("CourseList", ""))
            matches = code_pat.findall(course_list)
            
            for match in matches:
                prereq_course = norm_course(f"{match[0]} {match[1]}")
                if prereq_course != target:
                    or_group.add(prereq_course)
            
            if or_group:
                and_groups.append(or_group)
        
        if and_groups:
            prereq_groups[target] = and_groups

    return prereq_groups

def format_prerequisites(prereq_groups: dict[str, list[set[str]]], course_id: str) -> str:
    """Format prerequisites with correct AND/OR logic"""
    course_id_upper = course_id.upper()
    if course_id_upper not in prereq_groups or not prereq_groups[course_id_upper]:
        return ""
    
    groups = prereq_groups[course_id_upper]
    formatted_groups = []
    for group in groups:
        courses = sorted(group)
        if len(courses) == 1:
            formatted_groups.append(courses[0])
        else:
            formatted_groups.append("(" + " or ".join(courses) + ")")
    
    if len(formatted_groups) == 1:
        return formatted_groups[0]
    else:
        return " and ".join(formatted_groups)

def compute_prerequisite_depth(prereq_groups: dict[str, list[set[str]]]) -> dict[str, int]:
    """Compute topological depth of each course"""
    edges = defaultdict(set)
    all_courses = set()
    
    for target, groups in prereq_groups.items():
        all_courses.add(target)
        for group in groups:
            for prereq in group:
                all_courses.add(prereq)
                edges[prereq].add(target)
    
    depth = {course: 0 for course in all_courses}
    indegree = {course: 0 for course in all_courses}
    
    for target, groups in prereq_groups.items():
        for group in groups:
            for prereq in group:
                indegree[target] += 1
    
    queue = deque([course for course in all_courses if indegree[course] == 0])
    
    while queue:
        current = queue.popleft()
        for next_course in edges[current]:
            depth[next_course] = max(depth[next_course], depth[current] + 1)
            indegree[next_course] -= 1
            if indegree[next_course] == 0:
                queue.append(next_course)
    
    return depth

# ============================================================================
# DATA STORAGE CLASS
# ============================================================================

class CurriculumDataStore:
    """In-memory data store for curriculum information"""
    
    def __init__(self):
        self.courses_df: Optional[pd.DataFrame] = None
        self.majors_df: Optional[pd.DataFrame] = None
        self.prereq_groups: Dict[str, list[set[str]]] = {}
        self.prereq_depth: Dict[str, int] = {}
        self.curriculum_map: Optional[pd.DataFrame] = None
        
    def load_data(self, registrar_path: str, majors_path: str):
        """Load and process all data"""
        print(f"Loading registrar data from {registrar_path}")
        try:
            self.courses_df = pd.read_csv(registrar_path, encoding="latin1")
        except:
            self.courses_df = pd.read_csv(registrar_path, encoding="utf-8")
        
        print(f"Loading majors data from {majors_path}")
        ext = Path(majors_path).suffix.lower()
        if ext == ".csv":
            self.majors_df = pd.read_csv(majors_path)
        else:
            self.majors_df = pd.read_excel(majors_path, sheet_name="Sheet1", engine="openpyxl")
        
        print("Extracting prerequisites...")
        self.prereq_groups = extract_prereq_groups(self.courses_df)
        self.prereq_depth = compute_prerequisite_depth(self.prereq_groups)
        
        print("Building curriculum map...")
        self._build_curriculum_map()
        
        print(f"Data loaded: {len(self.courses_df)} courses, {len(self.prereq_groups)} with prerequisites")
    
    def _build_curriculum_map(self):
        """Build the full curriculum map"""
        majors_clean = self.majors_df.copy()
        
        def clean_course_id(course_str):
            if pd.isna(course_str):
                return ""
            course_str = str(course_str).strip()
            match = re.search(r'([A-Z]{2,4})\s*(\d{2,4}[A-Z]?)', course_str)
            if match:
                return f"{match.group(1)} {match.group(2)}"
            return course_str
        
        majors_clean["Course ID"] = majors_clean["Course"].apply(clean_course_id)
        
        term_mapping = {"FS": "Fall", "SS": "Spring", "US": "Summer", "WS": "Winter"}
        
        def clean_year(year_str):
            if pd.isna(year_str):
                return "Unknown"
            year_str = str(year_str).strip()
            for year in ["Freshman", "Sophomore", "Junior", "Senior"]:
                if year in year_str:
                    return year
            return year_str
        
        majors_clean["Year Level"] = majors_clean["Year"].apply(clean_year)
        majors_clean["Term Season"] = majors_clean["Term"].map(term_mapping).fillna("Unknown")
        
        courses = self.courses_df[["SUBJECT", "CRSE_CODE", "COURSE_TITLE_LONG"]].copy()
        courses["Course ID"] = [
            make_course_id(s, c) for s, c in zip(courses["SUBJECT"], courses["CRSE_CODE"])
        ]
        courses = courses[["Course ID", "COURSE_TITLE_LONG"]].drop_duplicates()
        courses = courses.rename(columns={"COURSE_TITLE_LONG": "Course Name"})
        
        curriculum = majors_clean.merge(courses, on="Course ID", how="left")
        curriculum["Course Name"] = curriculum["Course Name"].fillna("Unknown")
        
        curriculum["Prerequisites"] = [
            format_prerequisites(self.prereq_groups, str(cid)) 
            for cid in curriculum["Course ID"]
        ]
        curriculum["Prerequisite Depth"] = [
            self.prereq_depth.get(str(cid).upper(), 0) 
            for cid in curriculum["Course ID"]
        ]
        curriculum["Has Prerequisites"] = [
            str(cid).upper() in self.prereq_groups 
            for cid in curriculum["Course ID"]
        ]
        
        self.curriculum_map = curriculum

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="MSU Curriculum Map API",
    description="API for querying Michigan State University course data, prerequisites, and major requirements",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_store = CurriculumDataStore()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    registrar_path = "20250919_Registrars_Data(in).csv"
    majors_path = "CNS_Majors_Data.xlsx"  # Changed to CSV
    
    try:
        data_store.load_data(registrar_path, majors_path)
    except Exception as e:
        print(f"Warning: Could not load data on startup: {e}")
        print("API will return empty results until data is loaded")

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "message": "MSU Curriculum Map API",
        "data_loaded": data_store.courses_df is not None
    }

@app.get("/majors", response_model=List[Dict])
async def get_majors():
    """Get list of all available majors"""
    if data_store.curriculum_map is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    majors = data_store.curriculum_map[["Major", "Major Title", "Plan Title"]].drop_duplicates()
    return majors.to_dict('records')

@app.get("/majors/{major_code}", response_model=MajorPlan)
async def get_major_plan(major_code: str):
    """Get complete 4-year plan for a specific major"""
    if data_store.curriculum_map is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    major_data = data_store.curriculum_map[
        data_store.curriculum_map["Major"].astype(str) == str(major_code)
    ]
    
    if major_data.empty:
        raise HTTPException(status_code=404, detail=f"Major {major_code} not found")
    
    # FIX: Deduplicate courses before calculating credits
    seen_courses = {}
    courses = []
    total_credits = 0.0
    
    for _, row in major_data.iterrows():
        course_id = row["Course ID"]
        
        # Skip duplicates - keep first occurrence
        if course_id in seen_courses:
            continue
        
        seen_courses[course_id] = True
        
        # Parse credits
        try:
            credits_val = float(row.get("Credits", 0) or 0)
            total_credits += credits_val
        except:
            credits_val = 0.0
        
        courses.append(CurriculumCourse(
            course_id=course_id,
            course_name=row["Course Name"],
            year_level=row["Year Level"],
            term_season=row["Term Season"],
            credits=str(row.get("Credits", "")),
            prerequisites=row["Prerequisites"],
            prerequisite_depth=int(row["Prerequisite Depth"]),
            has_prerequisites=bool(row["Has Prerequisites"]),
            feasibility_warning=None
        ))
    
    return MajorPlan(
        major_code=major_code.upper(),
        major_title=major_data.iloc[0]["Major Title"],
        plan_title=major_data.iloc[0].get("Plan Title", ""),
        courses=courses,
        total_credits=round(total_credits, 1)
    )

@app.get("/courses/{course_id}", response_model=Course)
async def get_course(course_id: str):
    """Get details for a specific course"""
    if data_store.courses_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    course_id_clean = course_id.upper().replace("-", " ")
    
    match = re.match(r'([A-Z]{2,4})\s*(\d{2,4}[A-Z]?)', course_id_clean)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid course ID format")
    
    subject, code = match.groups()
    
    course_data = data_store.courses_df[
        (data_store.courses_df["SUBJECT"].str.upper() == subject) &
        (data_store.courses_df["CRSE_CODE"].astype(str).str.replace(".0", "") == code)
    ]
    
    if course_data.empty:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
    
    row = course_data.iloc[0]
    return Course(
        course_id=f"{subject} {code}",
        course_name=row["COURSE_TITLE_LONG"],
        subject=subject,
        course_code=code,
        credits=None
    )

@app.get("/courses/{course_id}/prerequisites", response_model=CoursePrerequisites)
async def get_course_prerequisites(course_id: str):
    """Get prerequisites for a specific course"""
    if data_store.prereq_groups is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    course_id_upper = course_id.upper().replace("-", " ")
    
    if course_id_upper not in data_store.prereq_groups:
        return CoursePrerequisites(
            course_id=course_id_upper,
            prerequisites=[],
            formatted="None",
            depth=0
        )
    
    groups = data_store.prereq_groups[course_id_upper]
    prereq_groups_list = [
        PrerequisiteGroup(courses=sorted(list(group)))
        for group in groups
    ]
    
    return CoursePrerequisites(
        course_id=course_id_upper,
        prerequisites=prereq_groups_list,
        formatted=format_prerequisites(data_store.prereq_groups, course_id_upper),
        depth=data_store.prereq_depth.get(course_id_upper, 0)
    )

@app.get("/graph", response_model=GraphData)
async def get_graph_data(
    major: Optional[str] = Query(None, description="Filter by major code"),
    max_depth: Optional[int] = Query(None, description="Maximum prerequisite depth")
):
    """Get graph data for visualization (nodes and edges)"""
    if data_store.curriculum_map is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = data_store.curriculum_map.copy()
    if major:
        df = df[df["Major"].astype(str) == str(major)]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Major {major} not found")
    
    nodes = []
    seen_courses = set()
    
    for _, row in df.iterrows():
        course_id = row["Course ID"]
        if course_id in seen_courses:
            continue
        seen_courses.add(course_id)
        
        depth = data_store.prereq_depth.get(course_id.upper(), 0)
        if max_depth and depth > max_depth:
            continue
        
        nodes.append(GraphNode(
            id=course_id,
            label=course_id,
            title=row["Course Name"],
            major=str(row.get("Major")) if pd.notna(row.get("Major")) else None,
            year=row.get("Year Level"),
            term=row.get("Term Season"),
            credits=str(row.get("Credits", "")),
            depth=depth
        ))
    
    edges = []
    course_ids = {node.id.upper() for node in nodes}
    
    for target, groups in data_store.prereq_groups.items():
        if target not in course_ids:
            continue
        for group in groups:
            for prereq in group:
                if prereq in course_ids:
                    edges.append(GraphEdge(
                        source=prereq,
                        target=target,
                        type="prerequisite"
                    ))
    
    return GraphData(nodes=nodes, edges=edges)

@app.get("/search")
async def search_courses(
    query: str = Query(..., min_length=2, description="Search term"),
    limit: int = Query(20, le=100, description="Maximum results")
):
    """Search for courses by ID or name"""
    if data_store.courses_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    query_upper = query.upper()
    
    matches = data_store.courses_df[
        (data_store.courses_df["SUBJECT"].str.upper().str.contains(query_upper, na=False)) |
        (data_store.courses_df["CRSE_CODE"].astype(str).str.contains(query_upper, na=False)) |
        (data_store.courses_df["COURSE_TITLE_LONG"].str.upper().str.contains(query_upper, na=False))
    ]
    
    results = []
    for _, row in matches.head(limit).iterrows():
        results.append({
            "course_id": make_course_id(row["SUBJECT"], row["CRSE_CODE"]),
            "course_name": row["COURSE_TITLE_LONG"],
            "subject": row["SUBJECT"]
        })
    
    return {"query": query, "count": len(results), "results": results}

@app.get("/courses/{course_id}/relations")
async def get_course_relations(course_id: str):

    if data_store.prereq_groups is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    cid = course_id.upper().replace("-", " ")

    if data_store.courses_df is not None:
        normalized_ids = {
            f"{row['SUBJECT']} {str(row['CRSE_CODE']).replace('.0','')}".upper()
            for _, row in data_store.courses_df.iterrows()
        }
        if cid not in normalized_ids:
            raise HTTPException(status_code=404, detail=f"Course {cid} does not exist in dataset")

    raw_groups = data_store.prereq_groups.get(cid, [])
    prereqs_raw = [sorted(list(group)) for group in raw_groups]
    prereqs_formatted = format_prerequisites(data_store.prereq_groups, cid) or "None"

    dependents = []
    for target, groups in data_store.prereq_groups.items():
        for group in groups:
            if cid in group:
                dependents.append(target)

    depth = data_store.prereq_depth.get(cid, 0)

    return {
        "course": cid,
        "prerequisites_raw": prereqs_raw,     
        "prerequisites_formatted": prereqs_formatted,
        "is_prereq_for": sorted(set(dependents)),
        "depth": depth
    }
