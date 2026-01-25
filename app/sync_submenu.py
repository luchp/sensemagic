"""
Sync FastAPI routers to WordPress menu as sub-menu items
Adds individual apps under "Articles" and "Applications" parent menu items
"""
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from wordpress_sync import WordPressSync
from shared.router_utils import discover_routers, format_router_title


def main():
    """Sync routers to WordPress menu as sub-menu items"""

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
    print("DISCOVERING ROUTERS")
    print("="*70)

    # Discover routers
    routers = discover_routers()
    print(f"\nFound {len(routers)} routers:")

    # Separate articles from other apps for display
    articles_router = routers.get('app_articles')
    other_routers = {k: v for k, v in routers.items() if k != 'app_articles'}

    if articles_router:
        # Discover articles to show in output
        from shared.article_utils import discover_articles
        articles_list = discover_articles(include_private=False)

        print(f"\n  Articles:")
        if articles_list:
            for article in articles_list[:3]:  # Show first 3 as example
                print(f"    - {article['title']}")
            if len(articles_list) > 3:
                print(f"    - ... and {len(articles_list) - 3} more")
        else:
            print(f"    - (no articles found)")

    if other_routers:
        print(f"\n  Applications:")
        for prefix in sorted(other_routers.keys()):
            app_title = format_router_title(prefix)
            print(f"    - {app_title} ({prefix})")

    print("\n" + "="*70)
    print("SYNCING AS SUB-MENU ITEMS")
    print("="*70)
    print("\nThis will add routers as child items under:")
    print("  - 'Articles' parent menu item")
    print("  - 'Applications' parent menu item")
    print("\nMenu items will be sorted by file modification date (newest first)")
    print("Make sure these parent items exist in your WordPress menu!")

    # Optional: specify custom parent titles
    # articles_parent = "My Articles Section"  # Change if your parent item has different name
    # applications_parent = "My Apps Section"   # Change if your parent item has different name

    # Sync to menu as sub-items
    success = wp_sync.sync_routers_as_submenu(
        routers,
        # menu_name="Main Menu",  # Uncomment to specify menu name
        # articles_parent_title=articles_parent,  # Uncomment to use custom parent names
        # applications_parent_title=applications_parent
    )

    if success:
        print("\n✓ Sub-menu sync completed successfully!")
        print(f"\nView your menu at: https://{domain}/wp-admin/nav-menus.php")
        print("\nExpected menu structure:")
        print("  Articles")

        # Show discovered articles in expected structure
        from shared.article_utils import discover_articles
        articles_list = discover_articles(include_private=False)
        if articles_list:
            for article in articles_list[:3]:
                print(f"    └─ {article['title']}")
            if len(articles_list) > 3:
                print(f"    └─ ... and {len(articles_list) - 3} more")

        print("  Applications")
        for prefix in sorted(other_routers.keys()):
            app_title = format_router_title(prefix)
            print(f"    └─ {app_title}")
        print("\n(Items sorted by file modification date, newest first)")
    else:
        print("\n✗ Sub-menu sync failed")
        print("\nTroubleshooting:")
        print("1. Ensure 'Articles' and 'Applications' menu items exist")
        print("2. Check that they link to the pages created by wordpress_sync.py")
        print("3. Verify WordPress permissions")
        sys.exit(1)


if __name__ == "__main__":
    main()

