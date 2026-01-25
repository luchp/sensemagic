#!/usr/bin/env python3
"""
WordPress Authentication Diagnostic Tool
Tests various authentication methods to identify the blocking issue
"""

import requests
import base64
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
BASE_URL = "https://sensemagic.nl"
WP_USER = "webmaster_sensemagic"
WP_APP_PASSWORD = "lMdo 1tTn Odr8 Mzz2 Slwq 8Pja"

print("="*70)
print("WordPress Authentication Diagnostic Tool")
print("="*70)
print()

# Test 1: Unauthenticated access
print("Test 1: Unauthenticated REST API access")
print("-" * 70)
try:
    response = requests.get(f"{BASE_URL}/wp-json/wp/v2/", verify=False, timeout=10)
    print(f"Status: {response.status_code}")
    if response.ok:
        print("✓ Unauthenticated access works!")
    else:
        print(f"✗ Failed: {response.status_code}")
        print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"✗ Error: {e}")
print()

# Test 2: Authenticated with Basic Auth header
print("Test 2: Authenticated access (Basic Auth header)")
print("-" * 70)
credentials = f"{WP_USER}:{WP_APP_PASSWORD}"
token = base64.b64encode(credentials.encode()).decode()
headers = {
    "Authorization": f"Basic {token}",
    "Content-Type": "application/json"
}
try:
    response = requests.get(f"{BASE_URL}/wp-json/wp/v2/users/me", 
                           headers=headers, verify=False, timeout=10)
    print(f"Status: {response.status_code}")
    if response.ok:
        user = response.json()
        print(f"✓ Authenticated as: {user.get('name', 'Unknown')}")
    else:
        print(f"✗ Failed: {response.status_code}")
        print(f"Response preview: {response.text[:300]}")
except Exception as e:
    print(f"✗ Error: {e}")
print()

# Test 3: Different endpoint (pages instead of users)
print("Test 3: List pages (authenticated)")
print("-" * 70)
try:
    response = requests.get(f"{BASE_URL}/wp-json/wp/v2/pages", 
                           headers=headers, verify=False, timeout=10)
    print(f"Status: {response.status_code}")
    if response.ok:
        pages = response.json()
        print(f"✓ Can list pages! Found {len(pages)} pages")
    else:
        print(f"✗ Failed: {response.status_code}")
        print(f"Response preview: {response.text[:300]}")
except Exception as e:
    print(f"✗ Error: {e}")
print()

# Test 4: Check response headers
print("Test 4: Check response headers")
print("-" * 70)
try:
    response = requests.get(f"{BASE_URL}/wp-json/wp/v2/users/me", 
                           headers=headers, verify=False, timeout=10)
    print(f"Status: {response.status_code}")
    print("Response Headers:")
    for key, value in response.headers.items():
        if key.lower() in ['server', 'x-powered-by', 'x-frame-options', 
                          'www-authenticate', 'x-robots-tag', 'link']:
            print(f"  {key}: {value}")
except Exception as e:
    print(f"✗ Error: {e}")
print()

# Test 5: Check if Application Passwords plugin is active
print("Test 5: Check WordPress plugins info")
print("-" * 70)
try:
    response = requests.get(f"{BASE_URL}/wp-json/", verify=False, timeout=10)
    if response.ok:
        data = response.json()
        print("WordPress API Info:")
        print(f"  Name: {data.get('name', 'N/A')}")
        print(f"  Description: {data.get('description', 'N/A')}")
        print(f"  URL: {data.get('url', 'N/A')}")
        if 'authentication' in data:
            print(f"  Authentication: {data.get('authentication')}")
except Exception as e:
    print(f"✗ Error: {e}")
print()

print("="*70)
print("Diagnostic Summary")
print("="*70)
print()
print("If Test 1 works but Test 2 fails with 403:")
print("  → WordPress is blocking authenticated requests")
print("  → Likely cause: Security plugin or .htaccess rules")
print()
print("Action needed:")
print("  1. Check WordPress → Plugins for security plugins")
print("  2. Check for .htaccess rules blocking Authorization header")
print("  3. Verify Application Passwords plugin is active and working")
print()

