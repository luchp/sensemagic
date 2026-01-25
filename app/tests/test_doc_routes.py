#!/usr/bin/env python
"""Test all documentation routes and check for broken links"""

import requests
from bs4 import BeautifulSoup
import re

BASE_URL = "http://localhost:8000"

# Test all documentation routes
routes = [
    "/app_rectifier/guide?standalone=true",
    "/app_rectifier/math?standalone=true",
    "/app_rectifier/wordpress?standalone=true",
]

print("Testing Documentation Routes and Links")
print("=" * 70)

for route in routes:
    url = BASE_URL + route
    print(f"\nTesting: {url}")
    
    try:
        response = requests.get(url, timeout=5)
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            # Parse HTML and find all links
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            
            internal_links = [
                link['href'] for link in links 
                if link['href'].startswith('/app_rectifier/')
            ]
            
            if internal_links:
                print(f"  Internal links found: {len(internal_links)}")
                for link in internal_links:
                    print(f"    → {link}")
                    
                    # Test each internal link
                    if not link.startswith('http'):
                        test_url = BASE_URL + link
                        test_response = requests.get(test_url, timeout=5)
                        status = "✓" if test_response.status_code == 200 else f"✗ ({test_response.status_code})"
                        print(f"      {status}")
            else:
                print("  No internal links found")
                
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 70)
print("\nSummary:")
print("If you see any ✗ marks above, those are broken links that need fixing.")
print("All ✓ marks mean the links work correctly.")

