"""Test script to verify the root route on Windows"""
import platform
from main import app

print(f"✓ Platform: {platform.system()}")
print(f"✓ Is Windows: {platform.system() == 'Windows'}")
print(f"\n✓ All routes in app:")
for route in app.routes:
    if hasattr(route, 'path'):
        methods = ", ".join(route.methods) if hasattr(route, 'methods') else "N/A"
        print(f"  {methods:15} {route.path}")

root_exists = any(r.path == "/" for r in app.routes if hasattr(r, 'path'))
print(f"\n✓ Root route (/) exists: {root_exists}")

if platform.system() == "Windows":
    print("\n✅ Running on Windows - Root route should be active!")
    print("   Access it at: http://localhost:8000/")
else:
    print("\n⚠️  Not running on Windows - Root route will not be added")

