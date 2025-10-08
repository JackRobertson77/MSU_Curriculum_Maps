#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict, deque
import pandas as pd

def read_registrar_csv(path: str) -> pd.DataFrame:
    """Safely reads the registrar data CSV file trying multiple encodings"""
    try:
        return pd.read_csv(path, encoding="latin1")
    except Exception:
        return pd.read_csv(path, encoding="utf-8", errors="replace")

def read_majors_excel(path: str) -> pd.DataFrame:
    """Reads the majors Excel spreadsheet (the curriculum plan file)"""
    return pd.read_excel(path, sheet_name="Sheet1")

def make_course_id(subj: str, code: str) -> str:
    """Combines subject and course code into a standard format like "CSE 331"""
    subj = str(subj).strip()
    num = str(code).strip()
    num = re.sub(r"\.0$", "", num)
    return f"{subj} {num}".strip()

def clean_course_id(course_str):
    """Normalizes course strings from the majors file (“CSE331”, “CSE 331”, “CSE-331”, etc.)"""
    if pd.isna(course_str):
        return ""
    course_str = str(course_str).strip()
    
    # Handle special patterns
    if any(x in course_str for x in ["Level", "XX", "or Higher", "or higher"]):
        return course_str
    
    # Extract standard course pattern
    match = re.search(r'([A-Z]{2,4})\s*(\d{2,4}[A-Z]?)', course_str)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return course_str

def extract_major_curriculum_structure(majors_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and structures the majors data to get each courses year level, term, and credits."""
    # Clean the data
    majors_clean = majors_df.copy()
    
    # Create proper course IDs
    majors_clean["Course ID"] = majors_clean["Course"].apply(clean_course_id)
    
    # Map term codes to seasons
    term_mapping = {
        "FS": "Fall",
        "SS": "Spring", 
        "US": "Summer",
        "WS": "Winter"
    }
    
    # Clean year values
    def clean_year(year_str):
        if pd.isna(year_str):
            return "Unknown"
        year_str = str(year_str).strip()
        if "Freshman" in year_str:
            return "Freshman"
        elif "Sophomore" in year_str:
            return "Sophomore" 
        elif "Junior" in year_str:
            return "Junior"
        elif "Senior" in year_str:
            return "Senior"
        return year_str
    
    majors_clean["Year Cleaned"] = majors_clean["Year"].apply(clean_year)
    majors_clean["Term Cleaned"] = majors_clean["Term"].map(term_mapping).fillna("Unknown")
    
    # Select relevant columns
    curriculum_structure = majors_clean[[
        "Major", "Plan Title", "Major Title", "Course ID", 
        "Year Cleaned", "Term Cleaned", "Credits", "Credits Min", "Fifth Year"
    ]].copy()
    
    curriculum_structure = curriculum_structure.rename(columns={
        "Year Cleaned": "Year Level",
        "Term Cleaned": "Term Season",
        "Credits Min": "Credits Minimum"
    })
    
    return curriculum_structure

def extract_prereq_groups(registrar_df: pd.DataFrame) -> dict[str, list[set[str]]]:
    """Builds prerequisite relationships from registrar data, preserving AND/OR logic.
    This is the core logic parser for prerequisites.
    The registrar file lists rules like “CSE 232 AND CSE 260” or “MTH 103 OR MTH 116”.
    This function:

    Groups data by course and requirement group.

    Detects “PRE” or “PREREQUISITE” rows.

    Extracts all course IDs that are required.

    Keeps them grouped in sets: each set is an AND group, and multiple sets mean OR options."""
    
    prereq_groups = defaultdict(list)
    code_pat = re.compile(r"\b([A-Z]{2,4})\s*(\d{2,4}[A-Z]?)\b")
    
    # Group by course and requirement group to understand the logic
    for course_id, group in registrar_df.groupby(["SUBJECT", "CRSE_CODE"]):
        target = make_course_id(course_id[0], course_id[1])
        
        # Group by RQRMNT_GROUP to understand prerequisite sets
        for req_group, req_group_data in group.groupby("RQRMNT_GROUP"):
            current_and_group = set()
            current_connect_type = None

            for _, row in req_group_data.iterrows():
                requisite_type = str(row.get("REQUISITE_TYPE", "")).strip().upper()
                rq_connect_type = str(row.get("RQ_CONNECT_TYPE", "")).strip().upper()
                rqdet_crse = str(row.get("RQDET_CRSE", "")).strip()
                descr254a = str(row.get("DESCR254A", "")).upper()
                
                # Check if this is a prerequisite row
                is_prereq_row = (
                    requisite_type == "PRE" or 
                    "PREREQUISITE:" in descr254a or
                    rqdet_crse and rqdet_crse != "NULL"
                )
                
                if not is_prereq_row:
                    continue
                
                # Extract prerequisite course
                prereq_course = None
                if rqdet_crse and rqdet_crse != "NULL":
                    match = code_pat.search(rqdet_crse)
                    if match:
                        prereq_course = f"{match.group(1)} {match.group(2)}"
                
                # Also check DESCR254A for additional prerequisites
                if not prereq_course and descr254a and "PREREQUISITE:" in descr254a:
                    matches = code_pat.findall(descr254a)
                    if matches:
                        subject, code = matches[0]  # Take first match
                        prereq_course = f"{subject} {code}"
                
                if prereq_course and prereq_course != target:
                    # Handle AND/OR logic
                    # This doesn't seem to be working correctly (MTH113 AND MTH113....)
                    if rq_connect_type == "AND":
                        current_and_group.add(prereq_course)
                    elif rq_connect_type == "OR":
                        # If we have an AND group, add it first (maybe this??)
                        if current_and_group:
                            prereq_groups[target].append(current_and_group.copy())
                            current_and_group.clear()
                        # Add the OR prerequisite as a single-item AND group
                        prereq_groups[target].append({prereq_course})
                    else:
                        # Default to AND if no connect type specified
                        current_and_group.add(prereq_course)
            
            # Add any remaining AND group
            if current_and_group:
                prereq_groups[target].append(current_and_group)
    
    return prereq_groups

def format_prerequisites(prereq_groups: dict[str, list[set[str]]], course_id: str) -> str:
    """Format prerequisites with AND/OR logic."""
    if course_id not in prereq_groups or not prereq_groups[course_id]:
        return ""
    
    groups = prereq_groups[course_id]
    formatted_groups = []
    
    for group in groups:
        if len(group) == 1:
            formatted_groups.append(next(iter(group)))  # Single course
        else:
            formatted_groups.append("(" + " AND ".join(sorted(group)) + ")")
    
    if len(formatted_groups) == 1:
        return formatted_groups[0]
    else:
        return " OR ".join(formatted_groups)

def compute_term_layers(prereq_groups: dict[str, list[set[str]]], course_ids: set[str]) -> dict[str, int]:
    """Compute topological ordering for courses with proper error handling."""
    # Convert prerequisite groups to simple edges for topological sort
    prereq_edges = defaultdict(set)
    for target, groups in prereq_groups.items():
        for group in groups:
            for prereq in group:
                prereq_edges[target].add(prereq)
    
    # Include ALL nodes that appear in the prerequisite graph
    all_nodes = set(course_ids)
    for target, prerequisites in prereq_edges.items():
        all_nodes.add(target)
        all_nodes.update(prerequisites)
    
    print(f"Debug: Processing {len(all_nodes)} total courses in prerequisite graph")
    print(f"Debug: {len(course_ids)} courses from majors data")
    
    # Build adjacency list and initialize indegree for ALL nodes
    adj = defaultdict(list)
    indeg = {node: 0 for node in all_nodes}
    
    # Build the graph
    for target, prerequisites in prereq_edges.items():
        for prereq in prerequisites:
            if prereq in all_nodes and target in all_nodes:
                adj[prereq].append(target)
                indeg[target] += 1
    
    # Initialize distances
    dist = {node: 0 for node in all_nodes}
    
    # Find nodes with no prerequisites
    q = deque([node for node, degree in indeg.items() if degree == 0])
    print(f"Debug: Starting with {len(q)} courses that have no prerequisites")
    
    # Process the graph
    while q:
        current = q.popleft()
        for neighbor in adj[current]:
            if dist[neighbor] < dist[current] + 1:
                dist[neighbor] = dist[current] + 1
            indeg[neighbor] -= 1
            if indeg[neighbor] == 0:
                q.append(neighbor)
    
    # Check for cycles
    remaining = [node for node, degree in indeg.items() if degree > 0]
    if remaining:
        print(f"Warning: {len(remaining)} courses may be in prerequisite cycles: {remaining[:10]}...")
    
    return {cid: dist.get(cid, 0) + 1 for cid in course_ids}

def build_comprehensive_curriculum_map(registrar_path: str, majors_path: str) -> pd.DataFrame:
    """Build complete curriculum map with better error handling."""
    # Read data
    registrar_df = read_registrar_csv(registrar_path)
    majors_df = read_majors_excel(majors_path)
    
    # Extract curriculum structure from majors data
    curriculum_structure = extract_major_curriculum_structure(majors_df)
    
    # Extract basic course info from registrar
    need = ["SUBJECT", "CRSE_CODE", "COURSE_TITLE_LONG"]
    for col in need:
        if col not in registrar_df.columns:
            raise KeyError(f"Missing column: {col}")
    
    courses = registrar_df[need].copy()
    courses["Course ID"] = [
        make_course_id(s, c) for s, c in zip(courses["SUBJECT"], courses["CRSE_CODE"])
    ]
    courses = courses[["Course ID", "COURSE_TITLE_LONG"]].drop_duplicates()
    courses = courses.rename(columns={"COURSE_TITLE_LONG": "Course Name"})
    
    # Extract prerequisites with AND/OR logic
    prereq_groups = extract_prereq_groups(registrar_df)
    
    # Debug: Show what prerequisites were found
    print(f"Found {len(prereq_groups)} courses with prerequisites")
    for course, groups in list(prereq_groups.items())[:5]:
        formatted = format_prerequisites(prereq_groups, course)
        print(f"  {course} requires: {formatted}")
        print(f"    Raw groups: {groups}")
    
    # Merge curriculum structure with course info
    merged = curriculum_structure.merge(
        courses, on="Course ID", how="left"
    )
    
    # Fill in missing course names
    merged["Course Name"] = merged["Course Name"].fillna("Course name not found in registrar data")
    
    # Get all course IDs from majors data
    all_course_ids = set(merged["Course ID"].dropna())
    print(f"Courses from majors data: {len(all_course_ids)}")
    
    # Compute prerequisite-based term ordering WITH ERROR HANDLING
    try:
        prereq_terms = compute_term_layers(prereq_groups, all_course_ids)
    except Exception as e:
        print(f"Error computing term layers: {e}")
        print("Falling back to default term assignment...")
        prereq_terms = {cid: 1 for cid in all_course_ids}
    
    # Create final curriculum map
    curriculum_map = pd.DataFrame({
        "Major Code": merged["Major"],
        "Major Title": merged["Major Title"],
        "Plan Title": merged["Plan Title"],
        "Course ID": merged["Course ID"],
        "Course Name": merged["Course Name"],
        "Year Level": merged["Year Level"],
        "Term Season": merged["Term Season"],
        "Fifth Year": merged["Fifth Year"],
        "Credits": merged["Credits"].fillna(""),
        "Credits Minimum": merged["Credits Minimum"].fillna(""),
        "Prerequisites": [format_prerequisites(prereq_groups, cid) for cid in merged["Course ID"]],
        "Prerequisite Term": [prereq_terms.get(cid, 1) for cid in merged["Course ID"]],
        "Has Prerequisites": [cid in prereq_groups and len(prereq_groups[cid]) > 0 for cid in merged["Course ID"]],
        "Is Prerequisite For": [sum(1 for groups in prereq_groups.values() 
                                  for group in groups if cid in group) for cid in merged["Course ID"]]
    })
    
    # Sort by major, year level, then prerequisite term
    year_order = {"Freshman": 1, "Sophomore": 2, "Junior": 3, "Senior": 4, "Unknown": 5}
    curriculum_map["Year Order"] = curriculum_map["Year Level"].map(year_order).fillna(5)
    
    curriculum_map = curriculum_map.sort_values([
        "Major Code", "Year Order", "Prerequisite Term", "Course ID"
    ]).drop(columns=["Year Order"])
    
    return curriculum_map.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Build comprehensive curriculum map with majors")
    ap.add_argument("--registrar", required=True, help="Path to registrar CSV")
    ap.add_argument("--majors", required=True, help="Path to majors Excel file")
    ap.add_argument("--out", required=True, help="Output path for curriculum table")
    
    args = ap.parse_args()
    
    # Build curriculum map
    curriculum_df = build_comprehensive_curriculum_map(args.registrar, args.majors)
    curriculum_df.to_csv(args.out, index=False)
    print(f"Curriculum map written: {args.out} ({len(curriculum_df)} rows)")
    
    # Display sample
    print("\nSample of curriculum map (showing major structure with AND/OR logic):")
    sample = curriculum_df.head(20)
    print(sample.to_string(index=False))
    
    # Display analytics
    print(f"\nCurriculum Analytics:")
    print(f"Total course offerings: {len(curriculum_df)}")
    print(f"Unique majors: {curriculum_df['Major Code'].nunique()}")
    print(f"Unique courses: {curriculum_df['Course ID'].nunique()}")
    print(f"Year level distribution:")
    print(curriculum_df['Year Level'].value_counts())

if __name__ == "__main__":
    main()
