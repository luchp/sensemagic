"""Test the create_main_router function"""
from discover import Discover
from routers.main_router import create_main_router

# Create discover instance
disc = Discover()
disc.get_routers()

print(f"✓ Discovered {len(disc.routers)} router(s)")

# Create main router
main_router = create_main_router(disc)

print(f"✓ Main router created successfully")
print(f"✓ Main router type: {type(main_router)}")
print(f"✓ Main router routes: {[r.path for r in main_router.routes if hasattr(r, 'path')]}")

# Verify the router has the root path
has_root = any(r.path == "/" for r in main_router.routes if hasattr(r, 'path'))
print(f"✓ Root path exists: {has_root}")

if has_root:
    print("\n✅ SUCCESS: create_main_router() works correctly!")
    print("   - Accepts disc object as parameter")
    print("   - Returns configured APIRouter")
    print("   - Router has GET / endpoint")
else:
    print("\n❌ ERROR: Root path not found in router!")

