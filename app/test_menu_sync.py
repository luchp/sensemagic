"""
Test script to demonstrate WordPress menu sync functionality
"""
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from wordpress_sync import WordPressSync
from shared.router_utils import discover_routers, format_router_title


def main():
    """Test syncing routers to WordPress menu"""

    # Load credentials
    try:
        import mysecrets
    except ImportError:
        print("ERROR: mysecrets.py not found. Create it with your WordPress credentials.")
        sys.exit(1)

    # Get credentials
    wp_user = mysecrets.WP_USER
    wp_password = mysecrets.WP_APP_PASSWORD
    domain = getattr(mysecrets, 'DOMAIN', 'sensemagic.nl')
    base_url = getattr(mysecrets, 'WP_BASE_URL', None)
    verify_ssl = getattr(mysecrets, 'WP_VERIFY_SSL', False)

    # Initialize WordPress sync
    wp_sync = WordPressSync(
        domain,
        wp_user,
        wp_password,
        base_url=base_url,
        verify_ssl=verify_ssl
    )

    # Test connection
    if not wp_sync.test_connection():
        print("Failed to connect to WordPress")
        sys.exit(1)

    print("\n" + "="*70)
    print("AVAILABLE MENUS")
    print("="*70)

    # List available menus
    menus = wp_sync.get_menus()
    if not menus:
        print("No menus found in WordPress.")
        print("\nTo create a menu:")
        print("1. Go to WordPress Admin → Appearance → Menus")
        print("2. Create a new menu (e.g., 'Main Menu', 'Primary', or 'Header')")
        print("3. Assign it to a theme location")
        sys.exit(1)

    for i, menu in enumerate(menus):
        print(f"{i+1}. {menu.get('name')} (ID: {menu['id']}, slug: {menu.get('slug')})")

    print("\n" + "="*70)
    print("DISCOVERING ROUTERS")
    print("="*70)

    # Discover routers
    routers = discover_routers()
    print(f"\nFound {len(routers)} routers:")
    for prefix in sorted(routers.keys()):
        app_title = format_router_title(prefix)
        print(f"  - {app_title} ({prefix})")

    print("\n" + "="*70)
    print("SYNCING TO MENU")
    print("="*70)

    print("\nChoose sync method:")
    print("1. Add routers as TOP-LEVEL menu items")
    print("2. Add routers as SUB-MENU items under 'Articles' and 'Applications'")
    print("3. Both")

    choice = input("\nEnter choice (1, 2, or 3) [default: 2]: ").strip() or "2"

    success = False

    if choice == "1":
        # Option 1: Top-level menu items
        print("\nAdding routers as top-level menu items...")
        success = wp_sync.sync_routers_to_menu(routers)

    elif choice == "2":
        # Option 2: Sub-menu items (recommended)
        print("\nAdding routers as sub-menu items...")
        print("Note: This requires existing 'Articles' and 'Applications' parent menu items")
        success = wp_sync.sync_routers_as_submenu(routers)

    elif choice == "3":
        # Both methods
        print("\nAdding routers as both top-level AND sub-menu items...")
        success1 = wp_sync.sync_routers_to_menu(routers)
        success2 = wp_sync.sync_routers_as_submenu(routers)
        success = success1 or success2

    else:
        print(f"\n✗ Invalid choice: {choice}")
        sys.exit(1)

    if success:
        print("\n✓ Menu sync completed successfully!")
        print(f"\nView your menu at: https://{domain}/wp-admin/nav-menus.php")
    else:
        print("\n✗ Menu sync failed")
        print("\nTroubleshooting:")
        print("- Ensure 'Articles' and 'Applications' menu items exist in WordPress")
        print("- Check WordPress permissions")
        sys.exit(1)


if __name__ == "__main__":
    main()

