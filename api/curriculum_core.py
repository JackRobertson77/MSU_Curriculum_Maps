#!/usr/bin/env python3
"""
MSU Curriculum Core API with Graph Visualization
"""

import pandas as pd
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Optional
import numpy as np

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
        _majors_df = pd.read_excel(
            majors_path, sheet_name="Sheet1", engine="openpyxl")

    if verbose:
        print("Extracting prerequisites...")

    _prereq_groups = _extract_prereq_groups(_courses_df)
    _prereq_depth = _compute_prerequisite_depth(_prereq_groups)

    if verbose:
        print(
            f"Loaded {len(_courses_df)} courses, {len(_prereq_groups)} with prerequisites")


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


def search_courses(query: str, limit: int = None) -> List[Dict]:
    """
    Search for courses by name or code.

    Args:
        query: Search term
        limit: Maximum number of results

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
        (_courses_df["COURSE_TITLE_LONG"].str.upper(
        ).str.contains(query_upper, na=False))
    ]
    matches = matches.drop_duplicates(subset=["SUBJECT", "CRSE_CODE"])
    if limit is not None:
        matches=matches.head(limit)

    results = {}
    for _, row in matches.iterrows():
        subject = row["SUBJECT"]
        number = str(row["CRSE_CODE"])
        course_id = _make_course_id(subject, number)

        results[course_id] = {
            "course_number": number,
            "subject": subject,
            "course_name": row["COURSE_TITLE_LONG"]
        }
    
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

    results = {}
    for _, row in matches.iterrows():
        subject = row["SUBJECT"]
        number = str(row["CRSE_CODE"])
        course_id = _make_course_id(subject, number)

        results[course_id] = {
            "course_number": number,
            "subject": subject,
            "course_name": row["COURSE_TITLE_LONG"]
        }
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

def adjacency_mat(major: Optional[str] = None, prereq_groups=None):
    """
    Produces an adjacency matrix based on prerequisite data.

    Args:
        major: optional arguement. Deafult will create adj mat for all courses. If specified, will create adj mat for just that major.
    Returns:
        2 variables returned:
            adj mat: numpy array of the adjacency matrix
            id: what the courses are for each row and column
    """
    global _majors_df, _prereq_groups

    if prereq_groups is None:
        prereq_groups = _prereq_groups
    if major is None:
        all_courses = set(prereq_groups.keys())
        for groups in prereq_groups.values():
            for group in groups:
                for prereq in group:
                    all_courses.add(prereq)
    else:
        if _majors_df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        major_df = _majors_df[_majors_df["Major"].astype(str) == str(major)]
        if major_df.empty:
            raise ValueError(f"Major not found: {major}")

        all_courses = set()
        for _, row in major_df.iterrows():
            course_str = str(row["Course"])
            match = re.search(r"([A-Z]{2,4})\s*(\d{2,4}[A-Z]?)", course_str)
            if match:
                cid = f"{match.group(1).upper()} {match.group(2)}"
                all_courses.add(cid)
        added = True
        while added:
            added = False
            for target, groups in prereq_groups.items():
                if target in all_courses:
                    for g in groups:
                        for prereq in g:
                            if prereq not in all_courses:
                                all_courses.add(prereq)
                                added = True
    filtered_groups = {
        c: [grp for grp in groups if any(p in all_courses for p in grp)]
        for c, groups in prereq_groups.items()
        if c in all_courses}

    all_courses = sorted(all_courses)
    index_map = {cid: i for i, cid in enumerate(all_courses)}

    N = len(all_courses)
    adj = np.zeros((N, N), dtype=int)

    for course, groups in prereq_groups.items():
        course_idx = index_map[course]
        for group in groups:
            for prereq in group:
                if prereq in index_map:
                    prereq_idx = index_map[prereq]
                    adj[prereq_idx, course_idx] = 1

    return adj, index_map

def adj_mat_as_pd_df(adj_mat, index_map):
    """
    Produces an adjacency matrix in pandas dataframe format.

    Args:
        adj_mat: 2x2 numpy array/matrix showing adjacency relationships. adj_mat output from adjacency_mat function is the intended input
        index_map: map of the course names based on index for the adjacency matrix. Intended input is id from adjacency_mat function
    Returns:
        adj_df: pandas dataframe of adjacency matrix with courses as row and column headers
    """
    course_names = list(index_map.keys())

    adj_df = pd.DataFrame(
        adj_mat,
        index=course_names,
        columns=course_names
    )
    return adj_df


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

    major_counts = _majors_df.groupby(
        ["Major", "Major Title"]).size().reset_index(name="courses")

    majors = []
    for _, row in major_counts.iterrows():
        majors.append({
            "name": row["Major Title"],
            "code": row["Major"],
            "courses": int(row["courses"])
        })

    return sorted(majors, key=lambda x: x["code"])


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

    # Add nodes with attributes
    for node in graph_data['nodes']:
        G.add_node(
            node['id'],
            title=node.get('title', ''),
            year=node.get('year', 'Unknown'),
            term=node.get('term', 'Unknown'),
            depth=node.get('depth', 0)
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

        for node_id in G.nodes():
            node_data = [n for n in graph_data['nodes']
                         if n['id'] == node_id][0]
            depth = node_data['depth']
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(node_id)

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
                    color_by: str = "year",
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

    # Determine node colors
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
            node_data = [n for n in graph_data['nodes']
                         if n['id'] == node_id][0]
            year = node_data.get('year', 'Unknown')
            node_colors.append(year_colors.get(year, '#9E9E9E'))

        legend_elements = [Patch(facecolor=color, label=year)
                           for year, color in year_colors.items()
                           if year in {n['year'] for n in graph_data['nodes']}]

    elif color_by == "depth":
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        depths = [n['depth'] for n in graph_data['nodes']]
        max_depth = max(depths) if depths else 1

        cmap = cm.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=0, vmax=max_depth)

        node_colors = []
        for node_id in G.nodes():
            node_data = [n for n in graph_data['nodes']
                         if n['id'] == node_id][0]
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
        major = graph_data['nodes'][0].get(
            'major', 'Unknown') if graph_data['nodes'] else 'Unknown'
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
        Dictionary with analysis results:
            - total_courses: Total number of courses
            - total_prerequisites: Total prerequisite relationships
            - courses_with_most_prereqs: List of (course_id, count) tuples
            - most_required_courses: List of (course_id, count) tuples
            - max_depth: Maximum prerequisite depth
            - avg_depth: Average prerequisite depth

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

    depths = [n['depth'] for n in graph_data['nodes']]

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

    print("="*70)
    print("CURRICULUM GRAPH ANALYSIS")
    print("="*70)

    print(f"\nOverview:")
    print(f"  Total Courses: {analysis['total_courses']}")
    print(f"  Total Prerequisites: {analysis['total_prerequisites']}")
    print(f"  Max Depth: {analysis['max_depth']}")
    print(f"  Avg Depth: {analysis['avg_depth']:.2f}")

    print(f"\nCourses with Most Prerequisites:")
    for course_id, count in analysis['courses_with_most_prereqs']:
        if count > 0:
            prereqs = get_prerequisites(course_id)
            print(f"  {course_id}: {count} prerequisites")
            print(f"    → {prereqs['formatted']}")

    print(f"\nMost Required Courses (as prerequisites):")
    for course_id, count in analysis['most_required_courses']:
        if count > 0:
            print(f"  {course_id}: required by {count} courses")

    print("="*70)


# ============================================================================
# DEMO/TEST FUNCTION
# ============================================================================

def demo():
    """
    Run a quick demonstration of the API including graph visualization.
    """
    print("="*70)
    print("MSU Curriculum Core API - Demo")
    print("="*70)

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
    print(f"{course['course_id']}: {course['course_name']}")

    print("\nCSE 232 prerequisites:")
    prereqs = get_prerequisites("CSE 232")
    print(f"{prereqs['formatted']}")
    print(f"Depth: {prereqs['depth']}")

    print("\nSearching for 'calculus' courses:")
    results = search_courses("calculus", limit=3)
    for r in results:
        print(f"   - {r['course_id']}: {r['course_name']}")

    print("\nGraph Visualization Demo:")
    print("Getting Chemistry BS graph data...")
    graph_data = get_graph_data(major="7105")
    print(f"Courses: {len(graph_data['nodes'])}")
    print(f"Prerequisites: {len(graph_data['edges'])}")

    print("\nAnalyzing graph...")
    print_graph_analysis(graph_data, top_n=3)

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
    
    graph = api.get_graph_data(major="7105")
    majors = api.get_major_list()
    
    VISUALIZATION FUNCTIONS (NEW!)
    
    # Create and visualize graphs
    G, pos = api.create_graph(graph_data, layout="hierarchical")
    fig, ax, G, pos = api.visualize_graph(
        graph_data, 
        layout="hierarchical",
        color_by="year",
        save_path="chemistry_curriculum.png"
    )
    
    # Analyze graphs
    analysis = api.analyze_graph(graph_data)
    api.print_graph_analysis(graph_data)
    
    EXAMPLES
    
    # Get prerequisites
    >>> prereqs = api.get_prerequisites("CSE 232")
    >>> print(prereqs['formatted'])
    (CMSE 202 or CSE 231) and (LB 118 or MTH 124 or MTH 132 or MTH 152H)
    
    # Visualize curriculum
    >>> graph_data = api.get_graph_data(major="7105")
    >>> api.visualize_graph(graph_data, layout="hierarchical", color_by="year")
    
    # Analyze curriculum structure
    >>> analysis = api.analyze_graph(graph_data)
    >>> print(f"Average prerequisite depth: {analysis['avg_depth']:.2f}")
    
    MORE HELP
    
    help(api.visualize_graph)       # Detailed help for visualization
    api.demo()                      # Run full demonstration
    """
    print(help_text)


__all__ = [
    'load_data',
    'get_course',
    'get_prerequisites',
    'get_dependent_courses',
    'search_courses',
    'get_courses_by_subject',
    'get_statistics',
    'get_graph_data',
    'get_major_list',
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
    print("\nFor quick help:")
    print("  api.help_quick()")
    print("\n" + "=" * 70)
