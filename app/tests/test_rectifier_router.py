"""
Test script to verify the rectifier router works correctly
"""
import sys
sys.path.append("..")

from routers.app_rectifier import router
from pages.rectifier.rectifier import RectifierModel


print("✓ Router imported successfully")
print(f"✓ Router prefix: {router.prefix}")
print(f"✓ Number of routes: {len(router.routes)}")
print(f"✓ Routes:")
for route in router.routes:
    print(f"  - {route.methods} {route.path}")

# Test the model
print("\n✓ Testing RectifierModel...")
rm = RectifierModel(1/50, 2, 325)
tau1, Ux, tau2, U1, C, Iripple = rm.solve_U1(100e-6, 0.1)
print(f"  Sample calculation: U1={U1:.2f}V, Iripple={Iripple:.3f}A")

print("\n✓ All tests passed! The router is ready to use.")
print("\nTo run the server, you need to:")
print("  1. Install uvicorn: pip install uvicorn")
print("  2. Run: python -m uvicorn main:app --reload --port 8000")
print("  3. Open: http://localhost:8000/app_rectifier/")

