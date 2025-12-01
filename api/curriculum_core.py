#!/usr/bin/env python3
"""
MSU Curriculum Core API with Graph Visualization
"""

import pandas as pd
import numpy as np
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
_forward_adj: Dict[str, List[str]] = {}
_backward_adj: Dict[str, List[Set[str]]] = {}

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


def _extract_number(code: str) -> int:
    """Extract numeric part of course code for sorting"""
    code_str = str(code).strip()
    code_str = re.sub(r"\.0$", "", code_str)
    match = re.match(r"(\d+)", code_str)
    return int(match.group(1)) if match else 0


def _extract_prereq_groups(
        registrar_df: pd.DataFrame) -> Dict[str, List[Set[str]]]:
    """
    Extract prerequisite relationships preserving AND/OR logic.
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
                    rq_connects = key_group['RQ_CONNECT'].astype(
                        str).str.upper().unique()
                    all_or = (len(rq_connects) == 1 and rq_connects[0] == 'OR')
                else:
                    all_or = False

                if all_or:
                    # All rows are OR alternatives combine into single group
                    or_group = set()
                    for _, row in key_group.iterrows():
                        course_list = str(row.get("CourseList", ""))
                        matches = code_pat.findall(course_list)
                        for match in matches:
                            prereq_course = _norm_course(
                                f"{match[0]} {match[1]}")
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
                            prereq_course = _norm_course(
                                f"{match[0]} {match[1]}")
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


def _build_adjacency_lists(prereq_groups: Dict[str, List[Set[str]]]):
    """Build forward and backward adjacency lists for O(1) lookups"""
    global _forward_adj, _backward_adj

    _backward_adj = prereq_groups.copy()

    _forward_adj = defaultdict(list)
    for course, prereq_list in prereq_groups.items():
        for group in prereq_list:
            for prereq in group:
                if course not in _forward_adj[prereq]:
                    _forward_adj[prereq].append(course)

    _forward_adj = dict(_forward_adj)


def _format_prerequisites(
        prereq_groups: Dict[str, List[Set[str]]], course_id: str) -> str:
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


def _compute_prerequisite_depth(
        prereq_groups: Dict[str, List[Set[str]]]) -> Dict[str, int]:
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

def load_data(
        registrar_path: str,
        majors_path: str,
        verbose: bool = True) -> None:
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
    except BaseException:
        _courses_df = pd.read_csv(registrar_path, encoding="utf-8")

    if verbose:
        print(f"Loading majors data from {majors_path}...")

    if not Path(majors_path).exists():
        raise FileNotFoundError(f"Majors file not found: {majors_path}")

    ext = Path(majors_path).suffix.lower()
    if ext == ".csv":
        _majors_df = pd.read_csv(majors_path)
    else:
        _majors_df = pd.read_excel(
            majors_path, sheet_name="Sheet1", engine="openpyxl")

    if verbose:
        print("Extracting prerequisites...")

    _prereq_groups = _extract_prereq_groups(_courses_df)
    _prereq_depth = _compute_prerequisite_depth(_prereq_groups)

    _build_adjacency_lists(_prereq_groups)

    if verbose:
        print(
            f"Loaded {
                len(_courses_df)} courses, {
                len(_prereq_groups)} with prerequisites")


def get_course(course_id: str) -> Dict:
    """
    Get information about a specific course.

    Args:
        course_id: Course ID (e.g., "CSE 232", "MTH-234")

    Returns:
        Dictionary containing course information (now as dict value, not in list)

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
        "name": row["COURSE_TITLE_LONG"],
        "subject": subject,
        "number": int(_extract_number(code)),
        "code": code
    }


def get_prerequisites(course_id: str) -> Dict[str, Dict]:
    """
    Get prerequisites for a course as dict-of-dicts.
    CHANGED: Now returns dict-of-dicts format like everything else

    Args:
        course_id: Course ID (e.g., "CSE 232")

    Returns:
        Dict where key=prerequisite course_id, value=course data

    Raises:
        RuntimeError: If data not loaded
    """
    if _prereq_groups is None or _courses_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")

    course_id_upper = course_id.upper().replace("-", " ")

    if course_id_upper not in _prereq_groups:
        return {}

    prereq_courses = {}
    for group in _prereq_groups[course_id_upper]:
        for prereq_id in group:
            if prereq_id not in prereq_courses:

                try:
                    match = re.match(
                        r'([A-Z]{2,4})\s*(\d{2,4}[A-Z]?)', prereq_id)
                    if match:
                        subject, code = match.groups()
                        course_data = _courses_df[
                            (_courses_df["SUBJECT"].str.upper() == subject) &
                            (_courses_df["CRSE_CODE"].astype(str).str.replace(".0", "") == code)
                        ]
                        if not course_data.empty:
                            row = course_data.iloc[0]
                            prereq_courses[prereq_id] = {
                                "name": row["COURSE_TITLE_LONG"],
                                "subject": subject,
                                "number": int(_extract_number(code))
                            }
                except BaseException:

                    prereq_courses[prereq_id] = {
                        "name": "Unknown",
                        "subject": prereq_id.split()[0] if ' ' in prereq_id else "",
                        "number": int(_extract_number(prereq_id.split()[-1])) if ' ' in prereq_id else 0
                    }

    return prereq_courses


def get_prerequisite_details(course_id: str) -> Dict:
    """
    Get detailed prerequisite information including AND/OR logic.

    Args:
        course_id: Course ID (e.g., "CSE 232")

    Returns:
        Dictionary with:
            - prerequisite_groups: List of groups (AND of ORs)
            - formatted: Human-readable string
            - depth: Prerequisite depth
            - total_count: Total number of prerequisite courses

    Raises:
        RuntimeError: If data not loaded
    """
    if _prereq_groups is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")

    course_id_upper = course_id.upper().replace("-", " ")

    if course_id_upper not in _prereq_groups:
        return {
            "prerequisite_groups": [],
            "formatted": "None",
            "depth": 0,
            "total_count": 0
        }

    groups = _prereq_groups[course_id_upper]
    prereq_list = [sorted(list(group)) for group in groups]

    all_prereqs = set()
    for group in groups:
        all_prereqs.update(group)

    return {
        "prerequisite_groups": prereq_list,
        "formatted": _format_prerequisites(_prereq_groups, course_id_upper),
        "depth": int(_prereq_depth.get(course_id_upper, 0)),
        "total_count": len(all_prereqs)
    }


def get_dependent_courses(course_id: str) -> Dict[str, Dict]:
    """
    Get courses that require this course as a prerequisite.
    NOW USES ADJACENCY LIST FOR O(1) LOOKUP!
    CHANGED: Returns dict-of-dicts like everything else

    Args:
        course_id: Course ID (e.g., "MTH 132")

    Returns:
        Dict where key=dependent course_id, value=course data

    Raises:
        RuntimeError: If data not loaded
    """
    if _forward_adj is None or _courses_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")

    course_id_upper = course_id.upper().replace("-", " ")

    dependent_ids = _forward_adj.get(course_id_upper, [])

    dependent_courses = {}
    for dep_id in dependent_ids:
        try:
            match = re.match(r'([A-Z]{2,4})\s*(\d{2,4}[A-Z]?)', dep_id)
            if match:
                subject, code = match.groups()
                course_data = _courses_df[
                    (_courses_df["SUBJECT"].str.upper() == subject) &
                    (_courses_df["CRSE_CODE"].astype(str).str.replace(".0", "") == code)
                ]
                if not course_data.empty:
                    row = course_data.iloc[0]
                    dependent_courses[dep_id] = {
                        "name": row["COURSE_TITLE_LONG"],
                        "subject": subject,
                        "number": int(_extract_number(code))
                    }
        except BaseException:

            dependent_courses[dep_id] = {
                "name": "Unknown",
                "subject": dep_id.split()[0] if ' ' in dep_id else "",
                "number": int(_extract_number(dep_id.split()[-1])) if ' ' in dep_id else 0
            }

    return dependent_courses


def get_all_courses() -> Dict[str, Dict]:
    """
    Get all courses as dict-of-dicts (CHANGED FROM LIST).

    Returns:
        Dict where key=course_id, value=course data

    Raises:
        RuntimeError: If data not loaded
    """
    if _courses_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")

    courses = {}
    for _, row in _courses_df[["SUBJECT", "CRSE_CODE",
                               "COURSE_TITLE_LONG"]].drop_duplicates().iterrows():
        course_id = _make_course_id(row["SUBJECT"], row["CRSE_CODE"])
        courses[course_id] = {
            "name": row["COURSE_TITLE_LONG"],
            "subject": row["SUBJECT"],

            "number": int(_extract_number(row["CRSE_CODE"]))
        }

    return courses


def search_courses(query: str, limit: int = 20) -> Dict[str, Dict]:
    """
    Search for courses by name or code.
    CHANGED: Returns dict-of-dicts instead of list

    Args:
        query: Search term
        limit: Maximum number of results (default: 20)

    Returns:
        Dict of matching courses where key=course_id

    Raises:
        RuntimeError: If data not loaded
    """
    if _courses_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")

    query_upper = query.upper()

    matches = _courses_df[
        (_courses_df["SUBJECT"].str.upper().str.contains(query_upper, na=False)) |
        (_courses_df["CRSE_CODE"].astype(str).str.contains(query_upper, na=False)) |
        (_courses_df["COURSE_TITLE_LONG"].str.upper(
        ).str.contains(query_upper, na=False))
    ]

    matches = matches.drop_duplicates(subset=["SUBJECT", "CRSE_CODE"])

    results = {}
    for _, row in matches.head(limit).iterrows():
        course_id = _make_course_id(row["SUBJECT"], row["CRSE_CODE"])
        results[course_id] = {
            "name": row["COURSE_TITLE_LONG"],
            "subject": row["SUBJECT"],
            "number": int(_extract_number(row["CRSE_CODE"]))
        }

    return results


def get_courses_by_subject(subject: str) -> Dict[str, Dict]:
    """
    Get all courses for a specific subject.
    CHANGED: Returns dict-of-dicts instead of list

    Args:
        subject: Subject code (e.g., "CSE", "MTH")

    Returns:
        Dict of courses where key=course_id

    Raises:
        RuntimeError: If data not loaded
    """
    if _courses_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")

    subject_upper = subject.upper()
    matches = _courses_df[_courses_df["SUBJECT"].str.upper() == subject_upper]

    results = {}
    for _, row in matches.iterrows():
        course_id = _make_course_id(row["SUBJECT"], row["CRSE_CODE"])
        results[course_id] = {
            "name": row["COURSE_TITLE_LONG"],
            "subject": row["SUBJECT"],
            "number": int(_extract_number(row["CRSE_CODE"]))
        }

    return results


def get_statistics() -> Dict:
    """
    Get statistics about loaded data.

    Returns:
        Dictionary with statistics (numpy ints converted to Python ints)

    Raises:
        RuntimeError: If data not loaded
    """
    if _courses_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")

    return {
        "total_courses": int(len(_courses_df)),
        "courses_with_prereqs": int(len(_prereq_groups)),
        "total_subjects": int(_courses_df["SUBJECT"].nunique()),
        "max_prereq_depth": int(max(_prereq_depth.values())) if _prereq_depth else 0
    }


def get_graph_data(
        major: Optional[str] = None,
        max_depth: Optional[int] = None) -> Dict:
    """
    Get graph data for visualization (nodes and edges).

    Args:
        major: Optional major code to filter by (e.g., "7105")
        max_depth: Optional maximum prerequisite depth to include

    Returns:
        Dictionary with 'nodes' (dict-of-dicts) and 'edges' lists

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

    nodes = {}
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
            (_courses_df["CRSE_CODE"].astype(
                str).str.replace(".0", "") == match.group(2))
        ]
        course_name = course_info.iloc[0]["COURSE_TITLE_LONG"] if not course_info.empty else "Unknown"

        year = str(row.get("Year", "Unknown"))
        term = str(row.get("Term", "Unknown"))

        for year_level in ["Freshman", "Sophomore", "Junior", "Senior"]:
            if year_level in year:
                year = year_level
                break

        nodes[course_id] = {
            "name": course_name,
            "major": major,
            "year": year,
            "term": term,
            "depth": int(depth)
        }

    edges = []
    course_ids = set(nodes.keys())

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


def get_major_list() -> Dict[str, Dict]:
    """
    Get list of all available majors.
    CHANGED: Returns dict-of-dicts with major names

    Returns:
        Dict where key=major_code, value=major info

    Raises:
        RuntimeError: If data not loaded
    """
    if _majors_df is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")

    major_counts = _majors_df.groupby(
        ["Major", "Description"]).size().reset_index(name="courses")

    majors = {}
    for _, row in major_counts.iterrows():
        major_code = str(row["Major"])
        major_name = str(row["Description"])
        majors[major_code] = {
            "name": f"Major: {major_name}",
            "courses": int(row["courses"])
        }

    return majors


# ============================================================================
# NEW FUNCTIONS
# ============================================================================

def detect_cycles() -> List[List[str]]:
    """
    Detect cycles in prerequisite graph using NetworkX.
    NEW FUNCTION!

    Returns:
        List of cycles (each cycle is a list of course IDs)

    Raises:
        RuntimeError: If data not loaded
        ImportError: If NetworkX not installed
    """
    if _prereq_groups is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")

    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX required. Install with: pip install networkx")

    # Build graph
    G = nx.DiGraph()
    for course, prereq_list in _prereq_groups.items():
        for group in prereq_list:
            for prereq in group:
                G.add_edge(prereq, course)

    # Find cycles
    try:
        cycles = list(nx.simple_cycles(G))
        return sorted(cycles, key=lambda x: (len(x), x[0]))
    except BaseException:
        return []


def get_bottleneck_courses(
        major: Optional[str] = None, top_n: int = 10) -> Dict[str, Dict]:
    """
    Find bottleneck courses - courses that block the most other courses.
    NEW FUNCTION!

    These are critical path courses that should be taken early.

    Args:
        major: Optional major code to filter by
        top_n: Number of top bottlenecks to return

    Returns:
        Dict where key=course_id, value={blocks: count, dependent_courses: [...]}

    Raises:
        RuntimeError: If data not loaded
    """
    if _forward_adj is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")

    # Get courses for major if specified
    if major:
        graph_data = get_graph_data(major=major)
        relevant_courses = set(graph_data["nodes"].keys())
    else:
        relevant_courses = set(_forward_adj.keys())

    # Count how many courses each prerequisite blocks
    bottlenecks = {}
    for course_id in relevant_courses:
        dependents = _forward_adj.get(course_id, [])
        if dependents:
            # Filter to only courses in the major
            if major:
                dependents = [d for d in dependents if d in relevant_courses]
            if dependents:
                bottlenecks[course_id] = {
                    "blocks": len(dependents),
                    "dependent_courses": sorted(dependents)
                }

    # Sort by number of courses blocked
    sorted_bottlenecks = dict(sorted(
        bottlenecks.items(),
        key=lambda x: x[1]["blocks"],
        reverse=True
    )[:top_n])

    return sorted_bottlenecks


def build_adjacency_mat(adj_list: Dict[str, List[str]] = None):
    """
    Build adjacency matrix in the form of a pandas dataframe to show prerequisite relationships.
    Assumes edges prereq -> course.
    Args:
        adj_list: 
    """

    #set default to _forward_adj list from above
    if adj_list is None:
        global _forward_adj
        adj_list = _forward_adj

    #get unique course names
    courses = sorted(set(adj_list.keys()) |
                     {c for deps in adj_list.values() for c in deps})

    #map out courses to an index
    index_map = {c: i for i, c in enumerate(courses)}

    #create empty matrix for adjacency matrix
    N = len(courses)
    mat = np.zeros((N, N), dtype=int)

    #fill with 1s based on a relationship and 0s
    for prereq, dependents in adj_list.items():
        for course in dependents:
            i = index_map[prereq]
            j = index_map[course]
            mat[i, j] = 1

    #return as a pandas dataframe to easily read and understand
    return pd.DataFrame(mat, index=courses, columns=courses)


# ============================================================================
# GRAPH VISUALIZATION FUNCTIONS
# ============================================================================

def create_graph(graph_data: Dict, layout: str = "spring", **kwargs):
    """
    Create a NetworkX graph from graph data.

    Args:
        graph_data: Dictionary from get_graph_data()
        layout: Layout algorithm - "spring", "hierarchical", or "circular"
        **kwargs: Additional arguments for layout (k, iterations, seed for spring)

    Returns:
        tuple: (G, pos) - NetworkX graph and position dictionary

    Requires:
        networkx (import networkx as nx)
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX is required. Install with: pip install networkx")

    G = nx.DiGraph()

    for course_id, node_data in graph_data['nodes'].items():
        G.add_node(
            course_id,
            title=node_data.get('name', ''),
            year=node_data.get('year', 'Unknown'),
            term=node_data.get('term', 'Unknown'),
            depth=node_data.get('depth', 0)
        )

    # Add edges
    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'])

    # Calculate layout
    if layout == "spring":
        k = kwargs.get('k', 3)
        iterations = kwargs.get('iterations', 50)
        seed = kwargs.get('seed', 42)
        pos = nx.spring_layout(G, k=k, iterations=iterations, seed=seed)

    elif layout == "hierarchical":
        # Position nodes by prerequisite depth
        pos = {}
        depth_groups = {}

        for course_id, node_data in graph_data['nodes'].items():
            depth = node_data['depth']
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(course_id)

        for depth, nodes in depth_groups.items():
            for i, node_id in enumerate(nodes):
                x = depth * 3
                y = i - len(nodes) / 2
                pos[node_id] = (x, y)

    elif layout == "circular":
        pos = nx.circular_layout(G)

    else:
        raise ValueError(
            f"Unknown layout: {layout}. Use 'spring', 'hierarchical', or 'circular'")

    return G, pos


def visualize_graph(graph_data: Dict,
                    layout: str = "spring",
                    color_by: str = "depth",
                    figsize: tuple = (20, 15),
                    title: Optional[str] = None,
                    save_path: Optional[str] = None,
                    show: bool = True,
                    **kwargs) -> tuple:
    """
    Visualize a curriculum graph using matplotlib and networkx.

    Args:
        graph_data: Dictionary from get_graph_data()
        layout: "spring", "hierarchical", or "circular"
        color_by: "year", "depth", or "uniform"
        figsize: Figure size (width, height)
        title: Graph title (auto-generated if None)
        save_path: Path to save figure (e.g., "graph.png")
        show: Whether to display the graph
        **kwargs: Additional layout arguments

    Returns:
        tuple: (figure, axis, graph, positions)

    Requires:
        matplotlib (import matplotlib.pyplot as plt)
        networkx (import networkx as nx)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        raise ImportError(
            "Matplotlib is required. Install with: pip install matplotlib")

    # Create graph
    G, pos = create_graph(graph_data, layout=layout, **kwargs)

    if color_by == "year":
        year_colors = {
            'Freshman': '#4CAF50',
            'Sophomore': '#2196F3',
            'Junior': '#FF9800',
            'Senior': '#F44336',
            'Unknown': '#9E9E9E'
        }
        node_colors = []
        for node_id in G.nodes():
            node_data = graph_data['nodes'][node_id]
            year = node_data.get('year', 'Unknown')
            node_colors.append(year_colors.get(year, '#9E9E9E'))

        legend_elements = [
            Patch(
                facecolor=color,
                label=year) for year,
            color in year_colors.items() if year in {
                n['year'] for n in graph_data['nodes'].values()}]

    elif color_by == "depth":
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        depths = [n['depth'] for n in graph_data['nodes'].values()]
        max_depth = max(depths) if depths else 1

        cmap = cm.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=0, vmax=max_depth)

        node_colors = []
        for node_id in G.nodes():
            node_data = graph_data['nodes'][node_id]
            depth = node_data.get('depth', 0)
            node_colors.append(cmap(norm(depth)))

        legend_elements = None

    else:  # uniform
        node_colors = 'lightblue'
        legend_elements = None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw graph
    import networkx as nx
    nx.draw(
        G, pos,
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        node_size=3000,
        font_size=8,
        font_weight='bold',
        arrows=True,
        arrowsize=20,
        edge_color='gray',
        linewidths=2
    )

    # Add title
    if title is None:
        if graph_data['nodes']:
            first_node = next(iter(graph_data['nodes'].values()))
            major = first_node.get('major', 'Unknown')
        else:
            major = 'Unknown'
        title = f"{major} - Prerequisite Flow ({layout.title()} Layout)"
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    # Add legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig, ax, G, pos


def analyze_graph(graph_data: Dict, top_n: int = 10) -> Dict:
    """
    Analyze a curriculum graph and return key statistics.

    Args:
        graph_data: Dictionary from get_graph_data()
        top_n: Number of top courses to return in rankings

    Returns:
        Dictionary with analysis results

    Requires:
        networkx (import networkx as nx)
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX is required. Install with: pip install networkx")

    G, _ = create_graph(graph_data, layout="spring")

    # Calculate statistics
    prereq_counts = {node: G.in_degree(node) for node in G.nodes()}
    dependent_counts = {node: G.out_degree(node) for node in G.nodes()}

    depths = [n['depth'] for n in graph_data['nodes'].values()]

    return {
        "total_courses": G.number_of_nodes(),
        "total_prerequisites": G.number_of_edges(),
        "courses_with_most_prereqs": sorted(
            prereq_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n],
        "most_required_courses": sorted(
            dependent_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n],
        "max_depth": max(depths) if depths else 0,
        "avg_depth": sum(depths) / len(depths) if depths else 0
    }


def print_graph_analysis(graph_data: Dict, top_n: int = 10):
    """
    Print a formatted analysis of a curriculum graph.

    Args:
        graph_data: Dictionary from get_graph_data()
        top_n: Number of top courses to show
    """
    analysis = analyze_graph(graph_data, top_n=top_n)

    print("=" * 70)
    print("CURRICULUM GRAPH ANALYSIS")
    print("=" * 70)

    print(f"\nOverview:")
    print(f"  Total Courses: {analysis['total_courses']}")
    print(f"  Total Prerequisites: {analysis['total_prerequisites']}")
    print(f"  Max Depth: {analysis['max_depth']}")
    print(f"  Avg Depth: {analysis['avg_depth']:.2f}")

    print(f"\nCourses with Most Prerequisites:")
    for course_id, count in analysis['courses_with_most_prereqs']:
        if count > 0:
            prereq_details = get_prerequisite_details(course_id)
            print(f"  {course_id}: {count} prerequisites")
            print(f"    → {prereq_details['formatted']}")

    print(f"\nMost Required Courses (as prerequisites):")
    for course_id, count in analysis['most_required_courses']:
        if count > 0:
            print(f"  {course_id}: required by {count} courses")

    print("=" * 70)


# ============================================================================
# DEMO/TEST FUNCTION
# ============================================================================

def demo():
    """
    Run a quick demonstration of the API including graph visualization.
    """
    print("=" * 70)
    print("MSU Curriculum Core API - Demo")
    print("=" * 70)

    print("\nLoading data...")
    load_data("20250919_Registrars_Data(in).csv",
              "CNS_Majors_Data.xlsx", verbose=False)
    print("Data loaded")

    print("\nStatistics:")
    stats = get_statistics()
    print(f"Total courses: {stats['total_courses']}")
    print(f"Courses with prerequisites: {stats['courses_with_prereqs']}")
    print(f"Total subjects: {stats['total_subjects']}")
    print(f"Max prerequisite depth: {stats['max_prereq_depth']}")

    print("\nGetting CSE 232 information:")
    course = get_course("CSE 232")
    print(f"CSE 232: {course['name']}")
    print(f"  Subject: {course['subject']}, Number: {course['number']}")

    print("\nCSE 232 prerequisites (dict-of-dicts):")
    prereqs = get_prerequisites("CSE 232")
    print(f"  Found {len(prereqs)} prerequisite courses:")
    for prereq_id, data in list(prereqs.items())[:3]:
        print(f"    - {prereq_id}: {data['name']}")

    # Get detailed info with AND/OR logic
    prereq_details = get_prerequisite_details("CSE 232")
    print(f"  Formatted: {prereq_details['formatted']}")
    print(f"  Depth: {prereq_details['depth']}")

    print("\nSearching for 'calculus' courses (now returns dict):")
    results = search_courses("calculus", limit=3)
    for course_id, data in results.items():
        print(f"   - {course_id}: {data['name']}")

    print("\nCycle detection:")
    cycles = detect_cycles()
    if cycles:
        print(f"  WARNING: Found {len(cycles)} prerequisite cycles!")
        for cycle in cycles[:3]:
            print(f"    {' -> '.join(cycle)}")
    else:
        print("  No cycles detected ✓")

    print("\nGraph Visualization Demo:")
    print("Getting Chemistry BS graph data...")
    graph_data = get_graph_data(major="7105")
    print(f"Courses: {len(graph_data['nodes'])} (now dict-of-dicts)")
    print(f"Prerequisites: {len(graph_data['edges'])}")

    print("\nBottleneck courses for major 7105:")
    bottlenecks = get_bottleneck_courses(major="7105", top_n=5)
    for course_id, data in bottlenecks.items():
        print(f"  {course_id}: blocks {data['blocks']} courses")

    print("\nAnalyzing graph...")
    print_graph_analysis(graph_data, top_n=3)

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


def help_quick():
    """
    Print quick help/usage guide.
    """
    help_text = """
    GETTING STARTED

    import curriculum_core as api

    # Load data (required first step)
    api.load_data("registrar.csv", "majors.xlsx")

    BASIC FUNCTIONS (ALL RETURN DICT-OF-DICTS!)

    course = api.get_course("CSE 232")  # Dict with course data
    prereqs = api.get_prerequisites("CSE 232")  # {"CSE 231": {...}, "MTH 132": {...}}
    prereq_details = api.get_prerequisite_details("CSE 232")  # AND/OR logic, depth
    dependents = api.get_dependent_courses("MTH 132")  # {"MTH 234": {...}, ...}
    results = api.search_courses("calculus")  # {"MTH 132": {...}, ...}
    stats = api.get_statistics()

    NEW FUNCTIONS

    cycles = api.detect_cycles()  # Find circular prerequisites
    bottlenecks = api.get_bottleneck_courses(major="7105")  # Critical path

    GRAPH FUNCTIONS

    graph = api.get_graph_data(major="7105")  # nodes now dict-of-dicts
    majors = api.get_major_list()  # {"7105": {...}, ...}

    EXAMPLES

    # Everything is dict-of-dicts now!
    >>> courses = api.get_all_courses()
    >>> cse232 = courses["CSE 232"]
    >>> print(cse232["number"])  # 232 as int for sorting

    # Prerequisites
    >>> prereqs = api.get_prerequisites("CSE 232")
    >>> for prereq_id, data in prereqs.items():
    ...     print(f"{prereq_id}: {data['name']}")

    # Get formatted prerequisite string
    >>> details = api.get_prerequisite_details("CSE 232")
    >>> print(details['formatted'])  # "(CSE 231 or CMSE 202) and MTH 132"

    # Dependents
    >>> deps = api.get_dependent_courses("MTH 132")
    >>> print(f"MTH 132 is needed by {len(deps)} courses")

    # Find bottlenecks
    >>> bottlenecks = api.get_bottleneck_courses(major="7105", top_n=5)
    >>> for course_id, data in bottlenecks.items():
    ...     print(f"{course_id} blocks {data['blocks']} courses")

    MORE HELP

    api.demo()
    """
    print(help_text)


__all__ = [
    'load_data',
    'get_course',
    'get_prerequisites',
    'get_prerequisite_details',
    'get_dependent_courses',
    'get_all_courses',
    'search_courses',
    'get_courses_by_subject',
    'get_statistics',
    'get_graph_data',
    'get_major_list',
    'detect_cycles',
    'get_bottleneck_courses',
    'create_graph',
    'visualize_graph',
    'analyze_graph',
    'print_graph_analysis',
    'demo',
    'help_quick'
]

if __name__ == "__main__":
    print("MSU Curriculum Core API with Graph Visualization")
    print("=" * 70)
    print("\nTo use this API:")
    print("  import curriculum_core as api")
    print("  api.load_data('registrar.csv', 'majors.xlsx')")
    print("  api.get_prerequisites('CSE 232')")
    print("\nTo visualize a curriculum:")
    print("  graph_data = api.get_graph_data(major='7105')")
    print("  api.visualize_graph(graph_data, layout='hierarchical')")
    print("\nTo see a demo:")
    print("  api.demo()")
    print("\nFor quick help:")
    print("  api.help_quick()")
    print("\n" + "=" * 70)
