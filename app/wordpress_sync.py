"""
WordPress synchronization module for FastAPI routers.
Uses WordPress REST API to create/update pages based on discovered routers.
"""

import requests
import base64
from typing import Dict, Optional
import sys
import urllib3
from pathlib import Path

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared.article_utils import discover_articles, extract_metadata_from_markdown
from shared.router_utils import (
    format_router_title,
    sort_routers_by_date,
    get_router_url,
    separate_articles_from_apps
)

# Disable SSL warnings when verify_ssl=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class WordPressSync:
    """Sync FastAPI routers to WordPress pages via REST API"""

    def __init__(self, domain: str, wp_user: str, wp_app_password: str,
                 base_url: str = None, verify_ssl: bool = None):
        """
        Initialize WordPress sync

        Parameters:
        domain: Domain name (e.g., "sensemagic.nl") - used for URLs in page content
        wp_user: WordPress username
        wp_app_password: WordPress application password (generated in WP admin)
        base_url: Base URL for WordPress API (default: https://{domain})
        verify_ssl: Verify SSL certificates (default: False for same-server connections)
        """
        self.domain = domain

        # Determine base URL
        if base_url:
            # Use explicitly provided base URL
            self.base_url = base_url.rstrip('/')
        else:
            # Default: use HTTPS with domain
            self.base_url = f"https://{domain}"

        # Determine SSL verification
        if verify_ssl is None:
            # Default: don't verify SSL (common for same-server or self-signed certs)
            self.verify_ssl = False
        else:
            self.verify_ssl = verify_ssl

        self.wp_url = f"{self.base_url}/wp-json/wp/v2"
        self.wp_user = wp_user
        self.wp_app_password = wp_app_password

        # Create authentication header
        credentials = f"{wp_user}:{wp_app_password}"
        token = base64.b64encode(credentials.encode()).decode()
        self.headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json"
        }

        print(f"WordPress API URL: {self.wp_url}")
        print(f"SSL Verification: {'enabled' if self.verify_ssl else 'disabled'}")

    def test_connection(self) -> bool:
        """Test WordPress API connection"""
        try:
            # Try to list pages - this tests authentication without accessing user info
            # which might be blocked by security plugins
            response = requests.get(
                f"{self.base_url}/wp-json/wp/v2/pages?per_page=1",
                headers=self.headers,
                verify=self.verify_ssl
            )
            if response.ok:
                print(f"✓ Connected to WordPress successfully!")
                print(f"  Can access pages endpoint - authentication working")
                return True
            else:
                # If pages endpoint fails, try users/me as fallback
                response = requests.get(
                    f"{self.base_url}/wp-json/wp/v2/users/me",
                    headers=self.headers,
                    verify=self.verify_ssl
                )
                if response.ok:
                    user_data = response.json()
                    print(f"✓ Connected to WordPress as: {user_data.get('name', 'Unknown')}")
                    return True
                else:
                    print(f"✗ WordPress connection failed: {response.status_code}", file=sys.stderr)
                    print(f"  Response: {response.text[:300]}", file=sys.stderr)
                    return False
        except Exception as e:
            print(f"✗ WordPress connection error: {e}", file=sys.stderr)
            return False

    def get_menus(self) -> list:
        """
        Get all WordPress menus

        Returns:
        List of menu dictionaries with id, name, slug
        """
        try:
            response = requests.get(
                f"{self.base_url}/wp-json/wp/v2/menus",
                headers=self.headers,
                verify=self.verify_ssl
            )
            if response.ok:
                menus = response.json()
                return menus
            else:
                print(f"✗ Failed to get menus: {response.status_code}", file=sys.stderr)
                print(f"  Response: {response.text[:300]}", file=sys.stderr)
                return []
        except Exception as e:
            print(f"✗ Error getting menus: {e}", file=sys.stderr)
            return []

    def get_menu_id_by_name(self, menu_name: str) -> Optional[int]:
        """
        Get menu ID by exact name match

        Args:
            menu_name: The exact name of the menu to find

        Returns:
            Menu ID or None if not found
        """
        menus = self.get_menus()
        if not menus:
            print("✗ No menus found in WordPress", file=sys.stderr)
            return None

        for menu in menus:
            if menu.get('name') == menu_name:
                print(f"✓ Found menu by name: {menu.get('name')} (ID: {menu['id']})")
                return menu['id']

        print(f"✗ Menu '{menu_name}' not found", file=sys.stderr)
        return None

    def get_primary_menu_id(self, menu_name: Optional[str] = None) -> Optional[int]:
        """
        Get the primary menu ID

        Args:
            menu_name: Optional menu name to match exactly. If provided, will search for this name.
                      If None, will auto-detect based on common slugs.

        Returns:
            Menu ID or None if no menus found
        """
        menus = self.get_menus()
        if not menus:
            print("✗ No menus found in WordPress", file=sys.stderr)
            return None

        # If menu_name is specified, search by exact name
        if menu_name:
            for menu in menus:
                if menu.get('name') == menu_name:
                    print(f"✓ Found menu by name: {menu.get('name')} (ID: {menu['id']})")
                    return menu['id']
            print(f"✗ Menu '{menu_name}' not found. Available menus:", file=sys.stderr)
            for menu in menus:
                print(f"  - {menu.get('name')}", file=sys.stderr)
            return None

        # Try to find primary menu by slug
        for menu in menus:
            if menu.get('slug') in ['primary', 'main', 'header']:
                print(f"✓ Found primary menu: {menu.get('name')} (ID: {menu['id']})")
                return menu['id']

        # Fall back to first menu
        first_menu = menus[0]
        print(f"✓ Using first available menu: {first_menu.get('name')} (ID: {first_menu['id']})")
        return first_menu['id']

    def add_custom_menu_item(self, menu_id: int, title: str, url: str,
                            parent_id: int = 0, order: int = 0) -> Optional[dict]:
        """
        Add custom link to WordPress menu using native REST API

        Args:
            menu_id: The ID of the menu (nav_menu term_id)
            title: Menu item display text
            url: Target URL
            parent_id: Parent menu item ID (0 for top-level)
            order: Menu order position

        Returns:
            Response dict with created menu item details or None on failure
        """
        endpoint = f"{self.base_url}/wp-json/wp/v2/menu-items"

        payload = {
            "title": title,
            "url": url,
            "menus": menu_id,  # Menu term ID
            "status": "publish",
            "type": "custom",  # Important: marks this as custom link
            "object": "custom",
            "menu_order": order
        }

        if parent_id > 0:
            payload["parent"] = parent_id

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=self.headers,
                verify=self.verify_ssl
            )

            if response.ok:
                item = response.json()
                print(f"✓ Added menu item: {title} (ID: {item['id']})")
                return item
            else:
                print(f"✗ Failed to add menu item {title}: {response.status_code}", file=sys.stderr)
                print(f"  Response: {response.text[:300]}", file=sys.stderr)
                return None
        except Exception as e:
            print(f"✗ Error adding menu item {title}: {e}", file=sys.stderr)
            return None

    def get_menu_items_by_title(self, menu_id: int, title: str) -> list:
        """
        Get menu items by title from a specific menu

        Args:
            menu_id: Menu ID to search in
            title: Title to search for

        Returns:
            List of matching menu items
        """
        try:
            response = requests.get(
                f"{self.base_url}/wp-json/wp/v2/menu-items",
                params={"menus": menu_id, "per_page": 100},
                headers=self.headers,
                verify=self.verify_ssl
            )
            if response.ok:
                all_items = response.json()
                # Normalize title for comparison (handle HTML entities, case differences)
                import html
                normalized_title = html.unescape(title).strip().lower()
                matching = []
                for item in all_items:
                    item_title = item.get('title', {}).get('rendered', '')
                    normalized_item_title = html.unescape(item_title).strip().lower()
                    if normalized_item_title == normalized_title:
                        matching.append(item)
                return matching
            return []
        except Exception as e:
            print(f"✗ Error searching menu items: {e}", file=sys.stderr)
            return []

    def get_menu_items_by_url(self, menu_id: int, url: str) -> list:
        """
        Get menu items by URL from a specific menu.
        More reliable than title matching for duplicate detection.

        Args:
            menu_id: Menu ID to search in
            url: URL to search for

        Returns:
            List of matching menu items
        """
        try:
            response = requests.get(
                f"{self.base_url}/wp-json/wp/v2/menu-items",
                params={"menus": menu_id, "per_page": 100},
                headers=self.headers,
                verify=self.verify_ssl
            )
            if response.ok:
                all_items = response.json()
                # Normalize URL for comparison
                normalized_url = url.strip().lower().rstrip('/')
                matching = []
                for item in all_items:
                    item_url = item.get('url', '')
                    normalized_item_url = item_url.strip().lower().rstrip('/')
                    if normalized_item_url == normalized_url:
                        matching.append(item)
                return matching
            return []
        except Exception as e:
            print(f"✗ Error searching menu items by URL: {e}", file=sys.stderr)
            return []

    def get_page_by_slug(self, slug: str) -> Optional[dict]:
        """
        Get WordPress page by slug

        Parameters:
        slug: Page slug

        Returns:
        Page data dictionary or None if not found
        """
        try:
            response = requests.get(
                f"{self.wp_url}/pages",
                params={"slug": slug},
                headers=self.headers,
                verify=self.verify_ssl
            )
            if response.ok:
                pages = response.json()
                return pages[0] if pages else None
            return None
        except Exception as e:
            print(f"Error getting page {slug}: {e}", file=sys.stderr)
            return None

    def create_page(self, title: str, slug: str, content: str, parent_id: int = 0) -> bool:
        """
        Create a new WordPress page

        Parameters:
        title: Page title
        slug: Page slug (URL-friendly name)
        content: Page content (HTML)
        parent_id: Parent page ID (0 for top-level)

        Returns:
        True if successful, False otherwise
        """
        page_data = {
            "title": title,
            "slug": slug,
            "content": content,
            "status": "publish",
            "parent": parent_id
        }

        try:
            response = requests.post(
                f"{self.wp_url}/pages",
                json=page_data,
                headers=self.headers,
                verify=self.verify_ssl
            )
            if response.ok:
                page = response.json()
                print(f"✓ Created WordPress page: {title} (ID: {page['id']}, slug: {slug})")
                return True
            else:
                print(f"✗ Failed to create page {title}: {response.status_code}", file=sys.stderr)
                print(f"  Response: {response.text}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"✗ Error creating page {title}: {e}", file=sys.stderr)
            return False

    def update_page(self, page_id: int, title: str, content: str) -> bool:
        """
        Update an existing WordPress page

        Parameters:
        page_id: WordPress page ID
        title: New page title
        content: New page content (HTML)

        Returns:
        True if successful, False otherwise
        """
        page_data = {
            "title": title,
            "content": content
        }

        try:
            response = requests.post(
                f"{self.wp_url}/pages/{page_id}",
                json=page_data,
                headers=self.headers,
                verify=self.verify_ssl
            )
            if response.ok:
                print(f"✓ Updated WordPress page: {title} (ID: {page_id})")
                return True
            else:
                print(f"✗ Failed to update page {page_id}: {response.status_code}", file=sys.stderr)
                print(f"  Response: {response.text}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"✗ Error updating page {page_id}: {e}", file=sys.stderr)
            return False

    def sync_routers_to_menu(self, routers: Dict, menu_id: Optional[int] = None,
                            menu_name: Optional[str] = None) -> bool:
        """
        Sync discovered FastAPI routers to WordPress menu as custom links

        Parameters:
        routers: Dictionary of router names to router objects
        menu_id: WordPress menu ID (if provided, takes precedence over menu_name)
        menu_name: WordPress menu name to search for (e.g., "Main Menu", "Primary Navigation")
                   Only used if menu_id is None

        Returns:
        True if sync successful, False otherwise
        """
        print(f"\nSyncing {len(routers)} routers to WordPress menu...")

        # Test connection first
        if not self.test_connection():
            print("✗ Cannot sync - WordPress connection failed", file=sys.stderr)
            return False

        # Get menu ID if not provided
        if menu_id is None:
            menu_id = self.get_primary_menu_id(menu_name=menu_name)
            if menu_id is None:
                print("✗ Cannot sync - No menu available", file=sys.stderr)
                return False

        success_count = 0

        # Sort routers by file modification date (newest first)
        sorted_routers = sort_routers_by_date(routers, reverse=True)

        # Add each router as a custom menu link (sorted by date)
        menu_order = 1
        for prefix, router, mtime in sorted_routers:
            app_title = format_router_title(prefix)
            url = get_router_url(self.domain, prefix, standalone=True)

            # Check if menu item already exists by URL (more reliable than title)
            existing_items = self.get_menu_items_by_url(menu_id, url)

            if existing_items:
                print(f"  ✓ Menu item for '{app_title}' already exists (URL match), skipping...")
            else:
                # Add new menu item with order
                result = self.add_custom_menu_item(menu_id, app_title, url, order=menu_order)
                if result:
                    success_count += 1
                    menu_order += 1

        print(f"\n{'='*60}")
        print(f"✓ Successfully added {success_count} menu items")
        print(f"{'='*60}\n")

        return success_count > 0

    def sync_routers_as_submenu(self, routers: Dict, menu_id: Optional[int] = None,
                                menu_name: Optional[str] = None,
                                articles_parent_title: str = "Articles",
                                applications_parent_title: str = "Applications") -> bool:
        """
        Sync discovered FastAPI routers to WordPress menu as child items under existing parent menu items

        This adds individual routers as sub-menu items under "Articles" and "Applications" parent items.

        Parameters:
        routers: Dictionary of router names to router objects
        menu_id: WordPress menu ID (if provided, takes precedence over menu_name)
        menu_name: WordPress menu name to search for (e.g., "Main Menu", "Primary Navigation")
        articles_parent_title: Title of the parent menu item for articles (default: "Articles")
        applications_parent_title: Title of the parent menu item for applications (default: "Applications")

        Returns:
        True if sync successful, False otherwise
        """
        print(f"\nSyncing {len(routers)} routers as sub-menu items...")

        # Test connection first
        if not self.test_connection():
            print("✗ Cannot sync - WordPress connection failed", file=sys.stderr)
            return False

        # Get menu ID if not provided
        if menu_id is None:
            menu_id = self.get_primary_menu_id(menu_name=menu_name)
            if menu_id is None:
                print("✗ Cannot sync - No menu available", file=sys.stderr)
                return False

        # Find parent menu items
        articles_parent_items = self.get_menu_items_by_title(menu_id, articles_parent_title)
        applications_parent_items = self.get_menu_items_by_title(menu_id, applications_parent_title)

        if not articles_parent_items:
            print(f"✗ Parent menu item '{articles_parent_title}' not found", file=sys.stderr)
            return False

        if not applications_parent_items:
            print(f"✗ Parent menu item '{applications_parent_title}' not found", file=sys.stderr)
            return False

        articles_parent_id = articles_parent_items[0]['id']
        applications_parent_id = applications_parent_items[0]['id']

        print(f"✓ Found parent: '{articles_parent_title}' (ID: {articles_parent_id})")
        print(f"✓ Found parent: '{applications_parent_title}' (ID: {applications_parent_id})")

        success_count = 0

        # Separate articles from other applications
        articles_router, other_routers = separate_articles_from_apps(routers)

        # Sort other routers by file modification date (newest first)
        sorted_routers = sort_routers_by_date(other_routers, reverse=True)

        # Add individual articles as children under Articles parent
        # Now supports hierarchical categories from subdirectories
        if articles_router:
            # Import articles_to_tree for hierarchical structure
            from shared.article_utils import articles_to_tree

            # Discover all articles and organize into tree
            articles_list = discover_articles(include_private=False)
            article_tree = articles_to_tree(articles_list)

            from pathlib import Path
            articles_dir = Path(__file__).parent / "pages" / "articles"

            # Track category menu items we create
            category_menu_items = {}

            # First, add category sub-menus under Articles parent
            category_order = 1
            for category_name, category_data in article_tree['categories'].items():
                category_slug = category_data.get('category_raw', '').lower().replace('_', '-')
                # Category menu item links to the articles page filtered by category (or just #)
                category_url = f"https://{self.domain}/app_articles/?category={category_slug}#"

                # Check if category menu item already exists
                existing_items = self.get_menu_items_by_title(menu_id, category_name)

                if existing_items:
                    # Use existing category item as parent
                    category_menu_items[category_name] = existing_items[0]['id']
                    print(f"  ✓ Category '{category_name}' already exists (ID: {existing_items[0]['id']})")
                else:
                    # Create new category menu item under Articles
                    print(f"  + Adding category: '{category_name}'")
                    result = self.add_custom_menu_item(
                        menu_id,
                        f"📁 {category_name}",  # Add folder emoji for visual hierarchy
                        category_url,
                        parent_id=articles_parent_id,
                        order=category_order
                    )
                    if result:
                        category_menu_items[category_name] = result['id']
                        success_count += 1
                        category_order += 1

            # Now add articles under their respective categories
            for category_name, category_data in article_tree['categories'].items():
                if category_name not in category_menu_items:
                    continue

                category_parent_id = category_menu_items[category_name]
                article_order = 1

                for article in category_data['articles']:
                    article_title = article['title']
                    article_slug = article['slug']
                    article_url = f"https://{self.domain}/app_articles/{article_slug}?standalone=true"

                    existing_items = self.get_menu_items_by_url(menu_id, article_url)

                    if existing_items:
                        print(f"    ✓ '{article_title}' already exists, skipping...")
                    else:
                        print(f"    + Adding: '{article_title}'")
                        result = self.add_custom_menu_item(
                            menu_id,
                            article_title,
                            article_url,
                            parent_id=category_parent_id,
                            order=article_order
                        )
                        if result:
                            success_count += 1
                            article_order += 1

            # Add uncategorized articles directly under Articles parent
            uncategorized_order = category_order + 100  # Start after categories
            for article in article_tree['uncategorized']:
                article_title = article['title']
                article_slug = article['slug']
                article_url = f"https://{self.domain}/app_articles/{article_slug}?standalone=true"

                existing_items = self.get_menu_items_by_url(menu_id, article_url)

                if existing_items:
                    print(f"  ✓ '{article_title}' already exists, skipping...")
                else:
                    print(f"  + Adding: '{article_title}'")
                    result = self.add_custom_menu_item(
                        menu_id,
                        article_title,
                        article_url,
                        parent_id=articles_parent_id,
                        order=uncategorized_order
                    )
                    if result:
                        success_count += 1
                        uncategorized_order += 1

        # Add other routers as children under Applications parent (sorted by date)
        menu_order = 1  # Start menu order counter
        for prefix, router, mtime in sorted_routers:
            app_title = format_router_title(prefix)
            url = get_router_url(self.domain, prefix, standalone=True)

            # Check if menu item already exists by URL (more reliable than title)
            existing_items = self.get_menu_items_by_url(menu_id, url)

            if existing_items:
                print(f"  ✓ Menu item for '{app_title}' already exists (URL match), skipping...")
            else:
                # Add new menu item as child of Applications with order
                result = self.add_custom_menu_item(menu_id, app_title, url,
                                                   parent_id=applications_parent_id,
                                                   order=menu_order)
                if result:
                    success_count += 1
                    menu_order += 1

        print(f"\n{'='*60}")
        print(f"✓ Successfully added {success_count} sub-menu items")
        print(f"{'='*60}\n")

        return success_count > 0

    def sync_routers_to_wordpress(self, routers: Dict, app_port: int = 9000) -> bool:
        """
        Sync discovered FastAPI routers to WordPress - creates two pages:
        1. Articles page listing all markdown articles
        2. Applications page listing all other apps

        Parameters:
        routers: Dictionary of router names to router objects
        app_port: FastAPI application port

        Returns:
        True if sync successful, False otherwise
        """
        print(f"\nSyncing {len(routers)} routers to WordPress...")

        # Test connection first
        if not self.test_connection():
            print("✗ Cannot sync - WordPress connection failed", file=sys.stderr)
            return False

        # Separate articles from other applications
        articles_router = routers.get('app_articles')
        other_routers = {k: v for k, v in routers.items() if k != 'app_articles'}

        success_count = 0

        # Create/update Articles page if app_articles exists
        if articles_router:
            if self._create_articles_page(articles_router):
                success_count += 1

        # Create/update Applications page with other routers
        if other_routers:
            if self._create_applications_page(other_routers):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"✓ Successfully synced {success_count} WordPress pages")
        print(f"{'='*60}\n")

        return success_count > 0

    def _create_articles_page(self, articles_router) -> bool:
        """Create/update the Articles page with list of articles, organized by category"""

        page_slug = "articles"
        page_title = "Articles"

        # Use shared article discovery utility
        articles_list = discover_articles(include_private=False)

        # Import and use articles_to_tree for hierarchical structure
        from shared.article_utils import articles_to_tree
        article_tree = articles_to_tree(articles_list)

        # Add file modification timestamps to articles
        from pathlib import Path
        articles_dir = Path(__file__).parent / "pages" / "articles"
        for article in articles_list:
            # Handle both flat and categorized articles
            if article.get('filepath'):
                md_file = articles_dir / article['filepath']
            else:
                md_file = articles_dir / article['filename']
            if md_file.exists():
                article['file_mtime'] = md_file.stat().st_mtime
            else:
                article['file_mtime'] = 0

        # Build articles page content
        articles_description = getattr(articles_router, 'description', None) or "Read our technical articles about algorithms, software, and electronics"

        page_content = f"""
<!-- wp:paragraph -->
<p>{articles_description}</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:html -->
<div id="articles-container">
<!-- /wp:html -->

"""
        # Add categorized articles first
        for category_name, category_data in article_tree['categories'].items():
            page_content += f"""
<!-- wp:heading {{"level":3}} -->
<h3>📁 {category_name}</h3>
<!-- /wp:heading -->

<!-- wp:html -->
<div class="category-articles" style="padding-left: 20px; border-left: 3px solid #2196F3; margin-bottom: 30px;">
<!-- /wp:html -->

"""
            for article in category_data['articles']:
                article_name = article['title']
                article_slug = article['slug']
                article_description = article['description']
                article_timestamp = article.get('file_mtime', 0)

                page_content += f"""
<!-- wp:paragraph {{"className":"article-item"}} -->
<p class="article-item" data-name="{article_name}" data-timestamp="{article_timestamp}" data-category="{category_name}"><a href="/app_articles/{article_slug}?standalone=true" target="_blank" rel="noopener"><strong>{article_name}</strong></a> - {article_description}</p>
<!-- /wp:paragraph -->

"""
            page_content += """
<!-- wp:html -->
</div>
<!-- /wp:html -->

"""

        # Add uncategorized articles
        if article_tree['uncategorized']:
            if article_tree['categories']:
                # Only show header if there are categories
                page_content += """
<!-- wp:heading {{"level":3}} -->
<h3>📄 General Articles</h3>
<!-- /wp:heading -->

"""
            for article in article_tree['uncategorized']:
                article_name = article['title']
                article_slug = article['slug']
                article_description = article['description']
                article_timestamp = article.get('file_mtime', 0)

                page_content += f"""
<!-- wp:paragraph {{"className":"article-item"}} -->
<p class="article-item" data-name="{article_name}" data-timestamp="{article_timestamp}" data-category="General"><a href="/app_articles/{article_slug}?standalone=true" target="_blank" rel="noopener"><strong>{article_name}</strong></a> - {article_description}</p>
<!-- /wp:paragraph -->

"""

        if not articles_list:
            page_content += """
<!-- wp:paragraph -->
<p>No articles available yet. Check back soon for new content!</p>
<!-- /wp:paragraph -->
"""

        page_content += """
<!-- wp:html -->
</div>
<!-- /wp:html -->
"""

        # Check if page already exists
        existing_page = self.get_page_by_slug(page_slug)

        if existing_page:
            # Update existing page
            print(f"Updating 'Articles' page (ID: {existing_page['id']})...")
            if self.update_page(existing_page['id'], page_title, page_content):
                print(f"✓ Successfully updated Articles page with {len(articles_list)} articles")
                return True
            else:
                print("✗ Failed to update Articles page", file=sys.stderr)
                return False
        else:
            # Create new page
            print("Creating 'Articles' page...")
            if self.create_page(page_title, page_slug, page_content):
                print(f"✓ Successfully created Articles page with {len(articles_list)} articles")
                return True
            else:
                print("✗ Failed to create Articles page", file=sys.stderr)
                return False

    def _create_applications_page(self, routers: Dict) -> bool:
        """Create/update the Applications page with all non-blog apps"""
        page_slug = "applications"
        page_title = "Applications"

        # Start with introduction
        page_content = """
<!-- wp:paragraph -->
<p>Browse our interactive web applications powered by FastAPI. Click any button to open the application in a new window.</p>
<!-- /wp:paragraph -->

<!-- wp:html -->
<div style="margin: 20px 0;">
    <label for="app-sort" style="font-weight: bold; margin-right: 10px;">Sort by:</label>
    <select id="app-sort" onchange="sortApps(this.value)" style="padding: 5px 10px; font-size: 14px;">
        <option value="name">Name (A-Z)</option>
        <option value="date">Date (Newest First)</option>
    </select>
</div>
<!-- /wp:html -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:html -->
<div id="apps-container">
<!-- /wp:html -->

"""

        # Add application info for each router
        for prefix, router in sorted(routers.items()):
            # Generate application metadata
            app_title = format_router_title(prefix)
            app_description = getattr(router, 'description', None) or f"Interactive {app_title} application"

            # Get file modification time for the router file
            from shared.router_utils import get_router_file_mtime
            file_timestamp = get_router_file_mtime(prefix)

            # Add application as clickable paragraph with data attributes
            page_content += f"""
<!-- wp:paragraph {{"className":"app-item"}} -->
<p class="app-item" data-name="{app_title}" data-timestamp="{file_timestamp}"><a href="/{prefix}/?standalone=true" target="_blank" rel="noopener"><strong>{app_title}</strong></a> - {app_description}</p>
<!-- /wp:paragraph -->

"""

        page_content += """
<!-- wp:html -->
</div>
<script>
function sortApps(sortBy) {
    const container = document.getElementById('apps-container');
    const apps = Array.from(container.querySelectorAll('.app-item'));
    
    apps.sort((a, b) => {
        if (sortBy === 'name') {
            const nameA = a.dataset.name.toLowerCase();
            const nameB = b.dataset.name.toLowerCase();
            return nameA.localeCompare(nameB);
        } else if (sortBy === 'date') {
            const tsA = parseFloat(a.dataset.timestamp || '0');
            const tsB = parseFloat(b.dataset.timestamp || '0');
            return tsB - tsA; // Newest first
        }
        return 0;
    });
    
    // Re-append in sorted order
    apps.forEach(app => container.appendChild(app));
}
</script>
<!-- /wp:html -->
"""

        # Check if page already exists
        existing_page = self.get_page_by_slug(page_slug)

        if existing_page:
            # Update existing page
            print(f"Updating 'Applications' page (ID: {existing_page['id']})...")
            if self.update_page(existing_page['id'], page_title, page_content):
                print(f"✓ Successfully updated Applications page with {len(routers)} applications")
                return True
            else:
                print("✗ Failed to update Applications page", file=sys.stderr)
                return False
        else:
            # Create new page
            print("Creating 'Applications' page...")
            if self.create_page(page_title, page_slug, page_content):
                print(f"✓ Successfully created Applications page with {len(routers)} applications")
                return True
            else:
                print("✗ Failed to create Applications page", file=sys.stderr)
                return False


def main():
    """Test WordPress sync functionality"""

    # Load credentials from mysecrets.py file - REQUIRED, no fallback
    try:
        import mysecrets
    except ImportError:
        print("\n" + "="*70, file=sys.stderr)
        print("ERROR: mysecrets.py file not found!", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print("WordPress sync requires app/mysecrets.py file with credentials.", file=sys.stderr)
        print("\nTo create it:", file=sys.stderr)
        print("  1. Copy mysecrets.py.example to mysecrets.py", file=sys.stderr)
        print("  2. Edit mysecrets.py and add your credentials:", file=sys.stderr)
        print("     - Generate application password in WordPress admin", file=sys.stderr)
        print("       (Users → Profile → Application Passwords)", file=sys.stderr)
        print("     - Set WP_USER and WP_APP_PASSWORD in mysecrets.py", file=sys.stderr)
        print("="*70 + "\n", file=sys.stderr)
        sys.exit(1)

    # Get required fields - no defaults
    if not hasattr(mysecrets, 'WP_USER'):
        print("ERROR: WP_USER not defined in mysecrets.py", file=sys.stderr)
        sys.exit(1)

    if not hasattr(mysecrets, 'WP_APP_PASSWORD'):
        print("ERROR: WP_APP_PASSWORD not defined in mysecrets.py", file=sys.stderr)
        sys.exit(1)

    wp_user = mysecrets.WP_USER
    wp_password = mysecrets.WP_APP_PASSWORD
    domain = getattr(mysecrets, 'DOMAIN', 'sensemagic.nl')
    base_url = getattr(mysecrets, 'WP_BASE_URL', None)
    verify_ssl = getattr(mysecrets, 'WP_VERIFY_SSL', False)

    # Validate credentials are not placeholders
    if not wp_user or wp_user == "admin":
        print("ERROR: WP_USER in mysecrets.py must be set to your actual WordPress username", file=sys.stderr)
        sys.exit(1)

    if not wp_password or wp_password.startswith("xxxx"):
        print("ERROR: WP_APP_PASSWORD in mysecrets.py must be set to your actual application password", file=sys.stderr)
        print("Generate one in WordPress admin: Users → Profile → Application Passwords", file=sys.stderr)
        sys.exit(1)

    # Test connection - fail immediately on error
    # Use configuration from mysecrets.py (base_url and verify_ssl)
    wp_sync = WordPressSync(
        domain,
        wp_user,
        wp_password,
        base_url=base_url,      # From mysecrets.py (optional)
        verify_ssl=verify_ssl   # From mysecrets.py (optional)
    )
    success = wp_sync.test_connection()

    if not success:
        print("\n" + "="*70, file=sys.stderr)
        print("ERROR: WordPress connection failed!", file=sys.stderr)
        print("="*70 + "\n", file=sys.stderr)
        sys.exit(1)

    # Test menu functionality
    print("\nTesting menu functionality:")
    print("-" * 60)

    # List available menus
    menus = wp_sync.get_menus()
    if menus:
        print(f"Found {len(menus)} menu(s):")
        for menu in menus:
            print(f"  - {menu.get('name')} (ID: {menu['id']}, slug: {menu.get('slug')})")
    else:
        print("No menus found - create a menu in WordPress admin first")

    print("-" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

