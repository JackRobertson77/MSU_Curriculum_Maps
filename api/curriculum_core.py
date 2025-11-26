#!/usr/bin/env python3
"""
MSU Curriculum Core API
"""

import pandas as pd
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Optional

# ============================================================================
# DATA STORAGE
# ============================================================================

_courses_df: Optional[pd.DataFrame] = None
_majors_df: Optional[pd.DataFrame] = None
_prereq_groups: Dict[str, List[Set[str]]] = {}
_prereq_depth: Dict[str, int] = {}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _make_course_id(subject: str, code: str) -> str:
    """Create standardized course ID from subject and code"""
    subject = str(subject).strip()
    code = str(code).strip()
    code = re.sub(r"\.0$", "", code)
    return f"{subject} {code}".strip()


def _norm_course(token: str) -> str:
    """Normalize course code to uppercase with single spaces"""
    token = str(token).strip()
    token = re.sub(r"\s+", " ", token)
    return token.upper()


def _extract_prereq_groups(registrar_df: pd.DataFrame) -> Dict[str, List[Set[str]]]:
    """
    Extract prerequisite relationships preserving AND/OR logic.
    
    This is the core algorithm that handles MSU's inconsistent data encoding.
    """
    prereq_groups = defaultdict(list)
    code_pat = re.compile(r"\b([A-Z]{2,4})\s+(\d{2,4}[A-Z]?)\b")

    for (subj, code), group in registrar_df.groupby(["SUBJECT", "CRSE_CODE"]):
        code_clean = re.sub(r"\.0$", "", str(code)).strip()
        target = _norm_course(f"{subj} {code_clean}")
        
        # Extract only PRE type (strict prerequisites)
        prereq_rows = group[
            (group.get("REQUISITE_TYPE", "").astype(str).str.upper() == "PRE") &
            (group.get("RQ_LINE_DET_TYPE", "").astype(str).str.upper() == "CLST") &
            (group["CourseList"].notna())
        ].copy()
        
        if prereq_rows.empty:
            continue

        and_groups = []
        
        # Group by RQ_LINE_KEY_NBR to identify prerequisite groups
        if 'RQ_LINE_KEY_NBR' in prereq_rows.columns:
            for line_key, key_group in prereq_rows.groupby('RQ_LINE_KEY_NBR'):
                # Check if all rows have RQ_CONNECT = "OR"
                if 'RQ_CONNECT' in key_group.columns:
                    rq_connects = key_group['RQ_CONNECT'].astype(str).str.upper().unique()
                    all_or = (len(rq_connects) == 1 and rq_connects[0] == 'OR')
                else:
                    all_or = False
                
                if all_or:
                    # All rows are OR alternatives - combine into single group
                    or_group = set()
                    for _, row in key_group.iterrows():
                        course_list = str(row.get("CourseList", ""))
                        matches = code_pat.findall(course_list)
                        for match in matches:
                            prereq_course = _norm_course(f"{match[0]} {match[1]}")
                            if prereq_course != target:
                                or_group.add(prereq_course)
                    if or_group:
                        and_groups.append(or_group)
                else:
                    # Rows are separate AND requirements
                    for _, row in key_group.iterrows():
                        or_group = set()
                        course_list = str(row.get("CourseList", ""))
                        matches = code_pat.findall(course_list)
                        for match in matches:
                            prereq_course = _norm_course(f"{match[0]} {match[1]}")
                            if prereq_course != target:
                                or_group.add(prereq_course)
                        if or_group:
                            and_groups.append(or_group)
        
        # Deduplication
        if and_groups:
            unique_groups = []
            seen = set()
            for group in and_groups:
                frozen = frozenset(group)
                if frozen not in seen:
                    seen.add(frozen)
                    unique_groups.append(group)
            prereq_groups[target] = unique_groups

    return dict(prereq_groups)


def _format_prerequisites(prereq_groups: Dict[str, List[Set[str]]], course_id: str) -> str:
    """Format prerequisites as human-readable string with AND/OR logic"""
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


def _compute_prerequisite_depth(prereq_groups: Dict[str, List[Set[str]]]) -> Dict[str, int]:
    """Compute prerequisite depth using topological sort"""
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
    
    return dict(depth)


# ============================================================================
# API FUNCTIONS
# ============================================================================

def load_data(registrar_path: str, majors_path: str, verbose: bool = True) -> None:
    """
    Load course and major data from files.
    
    This is the first function you should call to initialize the API.
    
    Args:
        registrar_path: Path to registrar CSV file (e.g., "registrar_data.csv")
        majors_path: Path to majors Excel/CSV file (e.g., "majors.xlsx")
        verbose: Print loading status messages (default: True)
    
    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If data is invalid
    """
    global _courses_df, _majors_df, _prereq_groups, _prereq_depth
    
    if verbose:
        print(f"Loading registrar data from {registrar_path}...")
    
    if not Path(registrar_path).exists():
        raise FileNotFoundError(f"Registrar file not found: {registrar_path}")
    
    try:
        _courses_df = pd.read_csv(registrar_path, encoding="latin1")
    except:
        _courses_df = pd.read_csv(registrar_path, encoding="utf-8")
    
    if verbose:
        print(f"Loading majors data from {majors_path}...")
    
    if not Path(majors_path).exists():
        raise FileNotFoundError(f"Majors file not found: {majors_path}")
    
    ext = Path(majors_path).suffix.lower()
    if ext == ".csv":
        _majors_df = pd.read_csv(majors_path)
    else:
        _majors_df = pd.read_excel(majors_path, sheet_name="Sheet1", engine="openpyxl")
    
    if verbose:
        print("Extracting prerequisites...")
    
    _prereq_groups = _extract_prereq_groups(_courses_df)
    _prereq_depth = _compute_prerequisite_depth(_prereq_groups)
    
    if verbose:
        print(f"Loaded {len(_courses_df)} courses, {len(_prereq_groups)} with prerequisites")


def get_course(course_id: str) -> Dict:
    """
    Get information about a specific course.
    
    Args:
        course_id: Course ID (e.g., "CSE 232", "MTH-234")
    
    Returns:
        Dictionary containing:
            - course_id: Standardized course ID
            - course_name: Full course name
            - subject: Subject code (e.g., "CSE")
            - course_code: Course number (e.g., "232")
    
    Raises:
        RuntimeError: If data not loaded (call load_data() first)
        ValueError: If course ID is invalid or course not found
    """
    if _courses_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")
    
    course_id = course_id.upper().replace("-", " ")
    match = re.match(r'([A-Z]{2,4})\s*(\d{2,4}[A-Z]?)', course_id)
    if not match:
        raise ValueError(f"Invalid course ID format: {course_id}")
    
    subject, code = match.groups()
    
    course_data = _courses_df[
        (_courses_df["SUBJECT"].str.upper() == subject) &
        (_courses_df["CRSE_CODE"].astype(str).str.replace(".0", "") == code)
    ]
    
    if course_data.empty:
        raise ValueError(f"Course not found: {course_id}")
    
    row = course_data.iloc[0]
    return {
        "course_id": f"{subject} {code}",
        "course_name": row["COURSE_TITLE_LONG"],
        "subject": subject,
        "course_code": code
    }


def get_prerequisites(course_id: str) -> Dict:
    """
    Get prerequisites for a course.
    
    Args:
        course_id: Course ID (e.g., "CSE 232")
    
    Returns:
        Dictionary containing:
            - course_id: Course ID
            - prerequisites: List of prerequisite groups (list of lists)
            - formatted: Human-readable prerequisite string
            - depth: Prerequisite depth (0 = no prerequisites)
    
    Raises:
        RuntimeError: If data not loaded
    
    Example:
        >>> prereqs = api.get_prerequisites("CSE 232")
        >>> print(prereqs['formatted'])
        (CMSE 202 or CSE 231) and (LB 118 or MTH 124 or MTH 132 or MTH 152H)
    """
    if _prereq_groups is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")
    
    course_id_upper = course_id.upper().replace("-", " ")
    
    if course_id_upper not in _prereq_groups:
        return {
            "course_id": course_id_upper,
            "prerequisites": [],
            "formatted": "None",
            "depth": 0
        }
    
    groups = _prereq_groups[course_id_upper]
    prereq_list = [sorted(list(group)) for group in groups]
    
    return {
        "course_id": course_id_upper,
        "prerequisites": prereq_list,
        "formatted": _format_prerequisites(_prereq_groups, course_id_upper),
        "depth": _prereq_depth.get(course_id_upper, 0)
    }


def get_dependent_courses(course_id: str) -> List[str]:
    """
    Get courses that require this course as a prerequisite.
    
    Args:
        course_id: Course ID (e.g., "MTH 132")
    
    Returns:
        List of course IDs that depend on this course
    
    Raises:
        RuntimeError: If data not loaded
    """
    if _prereq_groups is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")
    
    course_id_upper = course_id.upper().replace("-", " ")
    
    dependents = []
    for target, groups in _prereq_groups.items():
        for group in groups:
            if course_id_upper in group:
                dependents.append(target)
                break
    
    return sorted(set(dependents))


def get_all_courses() -> List[Dict]:
    """
    Get list of all courses.
    
    Returns:
        List of course dictionaries
    
    Raises:
        RuntimeError: If data not loaded
    """
    if _courses_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")
    
    courses = []
    for _, row in _courses_df[["SUBJECT", "CRSE_CODE", "COURSE_TITLE_LONG"]].drop_duplicates().iterrows():
        courses.append({
            "course_id": _make_course_id(row["SUBJECT"], row["CRSE_CODE"]),
            "course_name": row["COURSE_TITLE_LONG"],
            "subject": row["SUBJECT"]
        })
    
    return courses


def search_courses(query: str, limit: int = 20) -> List[Dict]:
    """
    Search for courses by name or code.
    
    Args:
        query: Search term
        limit: Maximum number of results (default: 20)
    
    Returns:
        List of matching course dictionaries
    
    Raises:
        RuntimeError: If data not loaded
    """
    if _courses_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")
    
    query_upper = query.upper()
    
    matches = _courses_df[
        (_courses_df["SUBJECT"].str.upper().str.contains(query_upper, na=False)) |
        (_courses_df["CRSE_CODE"].astype(str).str.contains(query_upper, na=False)) |
        (_courses_df["COURSE_TITLE_LONG"].str.upper().str.contains(query_upper, na=False))
    ]
    
    results = []
    for _, row in matches.head(limit).iterrows():
        results.append({
            "course_id": _make_course_id(row["SUBJECT"], row["CRSE_CODE"]),
            "course_name": row["COURSE_TITLE_LONG"],
            "subject": row["SUBJECT"]
        })
    
    return results


def get_courses_by_subject(subject: str) -> List[Dict]:
    """
    Get all courses for a specific subject.
    
    Args:
        subject: Subject code (e.g., "CSE", "MTH")
    
    Returns:
        List of course dictionaries
    
    Raises:
        RuntimeError: If data not loaded
    """
    if _courses_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")
    
    subject_upper = subject.upper()
    matches = _courses_df[_courses_df["SUBJECT"].str.upper() == subject_upper]
    
    results = []
    for _, row in matches.iterrows():
        results.append({
            "course_id": _make_course_id(row["SUBJECT"], row["CRSE_CODE"]),
            "course_name": row["COURSE_TITLE_LONG"],
            "subject": row["SUBJECT"]
        })
    
    return results


def get_statistics() -> Dict:
    """
    Get statistics about loaded data.
    
    Returns:
        Dictionary with statistics
    
    Raises:
        RuntimeError: If data not loaded
    """
    if _courses_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")
    
    return {
        "total_courses": len(_courses_df),
        "courses_with_prereqs": len(_prereq_groups),
        "total_subjects": _courses_df["SUBJECT"].nunique(),
        "max_prereq_depth": max(_prereq_depth.values()) if _prereq_depth else 0
    }


def get_graph_data(major: Optional[str] = None, max_depth: Optional[int] = None) -> Dict:
    """
    Get graph data for visualization (nodes and edges).
    
    Args:
        major: Optional major code to filter by (e.g., "CHEMBS")
        max_depth: Optional maximum prerequisite depth to include
    
    Returns:
        Dictionary with 'nodes' and 'edges' lists for graph visualization
        
    Raises:
        RuntimeError: If data not loaded
        ValueError: If major not found
    """
    if _majors_df is None or _courses_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")
    
    df = _majors_df.copy()
    if major:
        df = df[df["Major"].astype(str) == str(major)]
        if df.empty:
            raise ValueError(f"Major not found: {major}")
    
    nodes = []
    seen_courses = set()
    
    for _, row in df.iterrows():
        course_str = str(row.get("Course", "")).strip()
        match = re.search(r'([A-Z]{2,4})\s*(\d{2,4}[A-Z]?)', course_str)
        if not match:
            continue
        
        course_id = _make_course_id(match.group(1), match.group(2))
        
        if course_id in seen_courses:
            continue
        seen_courses.add(course_id)
        
        depth = _prereq_depth.get(course_id.upper(), 0)
        if max_depth is not None and depth > max_depth:
            continue
        
        course_info = _courses_df[
            (_courses_df["SUBJECT"].str.upper() == match.group(1).upper()) &
            (_courses_df["CRSE_CODE"].astype(str).str.replace(".0", "") == match.group(2))
        ]
        course_name = course_info.iloc[0]["COURSE_TITLE_LONG"] if not course_info.empty else "Unknown"
        
        year = str(row.get("Year", "Unknown"))
        term = str(row.get("Term", "Unknown"))
        
        for year_level in ["Freshman", "Sophomore", "Junior", "Senior"]:
            if year_level in year:
                year = year_level
                break
        
        nodes.append({
            "id": course_id,
            "label": course_id,
            "title": course_name,
            "major": major,
            "year": year,
            "term": term,
            "depth": depth
        })
    
    edges = []
    course_ids = {node["id"].upper() for node in nodes}
    
    for target, groups in _prereq_groups.items():
        if target not in course_ids:
            continue
        for group in groups:
            for prereq in group:
                if prereq in course_ids:
                    edges.append({
                        "source": prereq,
                        "target": target,
                        "type": "prerequisite"
                    })
    
    return {
        "nodes": nodes,
        "edges": edges
    }


def get_major_list() -> List[Dict]:
    """
    Get list of all available majors.
    
    Returns:
        List of major dictionaries with code and course count
    
    Raises:
        RuntimeError: If data not loaded
    """
    if _majors_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")
    
    major_counts = _majors_df.groupby("Major").size().reset_index(name="courses")
    
    majors = []
    for _, row in major_counts.iterrows():
        majors.append({
            "code": row["Major"],
            "courses": int(row["courses"])
        })
    
    return sorted(majors, key=lambda x: x["code"])


# ============================================================================
# DEMO/TEST FUNCTION
# ============================================================================

def demo():
    """
    Run a quick demonstration of the API.
    """
    print("="*70)
    print("MSU Curriculum Core API - Demo")
    print("="*70)
    
    print("\nLoading data...")
    load_data("20250919_Registrars_Data(in).csv", "CNS_Majors_Data.xlsx", verbose=False)
    print("Data loaded")
    
    print("\nStatistics:")
    stats = get_statistics()
    print(f"Total courses: {stats['total_courses']}")
    print(f"Courses with prerequisites: {stats['courses_with_prereqs']}")
    print(f"Total subjects: {stats['total_subjects']}")
    print(f"Max prerequisite depth: {stats['max_prereq_depth']}")
    
    print("\nGetting CSE 232 information:")
    course = get_course("CSE 232")
    print(f"{course['course_id']}: {course['course_name']}")
    
    print("\nCSE 232 prerequisites:")
    prereqs = get_prerequisites("CSE 232")
    print(f"{prereqs['formatted']}")
    print(f"Depth: {prereqs['depth']}")
    
    print("\nSearching for 'calculus' courses:")
    results = search_courses("calculus", limit=3)
    for r in results:
        print(f"   - {r['course_id']}: {r['course_name']}")
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


def help_quick():
    """
    Print quick help/usage guide.
    """
    help_text = """
    GETTING STARTED
    
    import curriculum_core as api
    
    # Load data (required first step)
    api.load_data("registrar.csv", "majors.xlsx")
    
    BASIC FUNCTIONS
    
    course = api.get_course("CSE 232")
    prereqs = api.get_prerequisites("CSE 232")
    dependents = api.get_dependent_courses("MTH 132")
    results = api.search_courses("calculus")
    stats = api.get_statistics()
    
    GRAPH FUNCTIONS
    
    graph = api.get_graph_data(major="CHEMBS")
    majors = api.get_major_list()
    
    EXAMPLES
    
    # Get prerequisites
    >>> prereqs = api.get_prerequisites("CSE 232")
    >>> print(prereqs['formatted'])
    (CMSE 202 or CSE 231) and (LB 118 or MTH 124 or MTH 132 or MTH 152H)
    
    # Get dependent courses
    >>> dependents = api.get_dependent_courses("MTH 132")
    >>> print(f"MTH 132 is required by {len(dependents)} courses")
    
    # Get graph data
    >>> graph = api.get_graph_data(major="7105")
    >>> print(f"Chemistry major: {len(graph['nodes'])} courses")
    
    MORE HELP
    
    help(api.get_prerequisites)    # Detailed help for one function
    api.demo()                      # Run full demonstration
    """
    print(help_text)


__all__ = [
    'load_data',
    'get_course',
    'get_prerequisites',
    'get_dependent_courses',
    'get_all_courses',
    'search_courses',
    'get_courses_by_subject',
    'get_statistics',
    'get_graph_data',
    'get_major_list',
    'demo',
    'help_quick'
]


if __name__ == "__main__":
    print("MSU Curriculum Core API")
    print("=" * 70)
    print("\nTo use this API:")
    print("  import curriculum_core as api")
    print("  api.load_data('registrar.csv', 'majors.xlsx')")
    print("  api.get_prerequisites('CSE 232')")
    print("\nTo see a demo:")
    print("  api.demo()")
    print("\nFor quick help:")
    print("  api.help_quick()")
    print("\n" + "=" * 70)
