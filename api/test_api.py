#!/usr/bin/env python3
"""
Quick test script for curriculum_core API

Run this to verify the API works correctly.
"""

import curriculum_core as api

print("="*70)
print("Testing MSU Curriculum Core API")
print("="*70)

# Test 1: Load data
print("\n[TEST 1] Loading data...")
try:
    api.load_data("20250919_Registrars_Data(in).csv", "CNS_Majors_Data.xlsx", verbose=False)
    print("PASS - Data loaded")
except Exception as e:
    print(f"FAIL - {e}")
    exit(1)

# Test 2: Get statistics
print("\n[TEST 2] Getting statistics...")
try:
    stats = api.get_statistics()
    print(f"PASS - {stats['total_courses']} courses, {stats['courses_with_prereqs']} with prereqs")
except Exception as e:
    print(f"FAIL - {e}")
    exit(1)

# Test 3: Get course
print("\n[TEST 3] Getting CSE 232 info...")
try:
    course = api.get_course("CSE 232")
    assert course['course_id'] == "CSE 232"
    print(f"PASS - {course['course_name']}")
except Exception as e:
    print(f"FAIL - {e}")
    exit(1)

# Test 4: Get prerequisites
print("\n[TEST 4] Getting CSE 232 prerequisites...")
try:
    prereqs = api.get_prerequisites("CSE 232")
    expected = "(CMSE 202 or CSE 231) and (LB 118 or MTH 124 or MTH 132 or MTH 152H)"
    assert prereqs['formatted'] == expected, f"Got: {prereqs['formatted']}"
    print(f"PASS - {prereqs['formatted']}")
except Exception as e:
    print(f"FAIL - {e}")
    exit(1)

# Test 5: Search
print("\n[TEST 5] Searching for 'calculus'...")
try:
    results = api.search_courses("calculus", limit=5)
    assert len(results) > 0
    print(f"PASS - Found {len(results)} courses")
except Exception as e:
    print(f"FAIL - {e}")
    exit(1)

# Test 6: Multiple courses
print("\n[TEST 6] Testing multiple courses...")
test_cases = {
    "CSE 232": "(CMSE 202 or CSE 231) and (LB 118 or MTH 124 or MTH 132 or MTH 152H)",
    "MTH 234": "(LB 119 or MTH 133 or MTH 153H)",
    "MTH 132": "(MTH 103 or MTH 114) and (MTH 103B or MTH 114)",
}

all_passed = True
for course_id, expected in test_cases.items():
    prereqs = api.get_prerequisites(course_id)
    if prereqs['formatted'] == expected:
        print(f"{course_id}")
    else:
        print(f" {course_id}")
        print(f"     Expected: {expected}")
        print(f"     Got: {prereqs['formatted']}")
        all_passed = False

if all_passed:
    print("PASS - All courses correct")
else:
    print("FAIL - Some courses incorrect")
    exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nThe API is working correctly. You can now:")
print("1. Use it in Python: import curriculum_core as api")
print("2. Try the demo: python curriculum_core.py")
print("3. Open API_Demo.ipynb in Jupyter")
print("4. (Optional) Start web API: uvicorn curriculum_api:app --reload")
