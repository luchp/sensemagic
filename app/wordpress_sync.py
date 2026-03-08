"""
WordPress synchronization module for FastAPI routers.
Uses WP-CLI (command-line) to create/update pages and menus,
bypassing Apache/nginx entirely.
"""

import json
import subprocess
import base64
from typing import Dict, List, Optional
import sys
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


class WordPressSync:
    """Sync FastAPI routers to WordPress pages via WP-CLI"""

    # WP-CLI configuration
    WP_PATH = "/var/www/vhosts/sensemagic.nl/httpdocs"
    PHP_BIN = "/opt/plesk/php/8.3/bin/php"
    WP_CLI = "/usr/local/bin/wp"

    def __init__(self, domain: str, wp_user: str = None, wp_app_password: str = None,
                 base_url: str = None, verify_ssl: bool = None):
        """
        Initialize WordPress sync

        Parameters:
        domain: Domain name (e.g., "sensemagic.nl") - used for URLs in page content
        wp_user: WordPress username (kept for API compatibility, not used by WP-CLI)
        wp_app_password: WordPress application password (kept for API compatibility)
        base_url: Base URL (kept for API compatibility)
        verify_ssl: SSL verification (kept for API compatibility)
        """
        self.domain = domain
        self.wp_user = wp_user

    def _wp(self, *args, input_data: str = None) -> subprocess.CompletedProcess:
        """Run a WP-CLI command and return the CompletedProcess result."""
        cmd = [
            self.PHP_BIN, self.WP_CLI,
            f"--path={self.WP_PATH}",
            "--allow-root",
        ] + list(args)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input=input_data,
            timeout=30,
        )

    def _wp_json(self, *args) -> Optional[any]:
        """Run a WP-CLI command with --format=json and return parsed JSON, or None on error."""
        result = self._wp(*args, "--format=json")
        if result.returncode != 0:
            print(f"✗ WP-CLI error: {result.stderr.strip()}", file=sys.stderr)
            return None
        try:
            return json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            print(f"✗ WP-CLI returned non-JSON: {result.stdout[:200]}", file=sys.stderr)
            return None

    def test_connection(self) -> bool:
        """Test WP-CLI can talk to WordPress"""
        result = self._wp("post", "list", "--post_type=page", "--posts_per_page=1",
                          "--fields=ID", "--format=json")
        if result.returncode == 0:
            print("✓ Connected to WordPress via WP-CLI")
            return True
        print(f"✗ WP-CLI connection failed: {result.stderr.strip()}", file=sys.stderr)
        return False

    # ------------------------------------------------------------------
    # Menu helpers
    # ------------------------------------------------------------------

    def get_menus(self) -> list:
        """Get all WordPress menus."""
        result = self._wp("menu", "list", "--format=json")
        if result.returncode != 0:
            print(f"✗ Failed to get menus: {result.stderr.strip()}", file=sys.stderr)
            return []
        try:
            menus = json.loads(result.stdout)
            # WP-CLI returns: term_id, name, slug, count
            return [{"id": int(m["term_id"]), "name": m["name"], "slug": m["slug"]} for m in menus]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"✗ Error parsing menus: {e}", file=sys.stderr)
            return []

    def get_menu_id_by_name(self, menu_name: str) -> Optional[int]:
        """Get menu ID by exact name match."""
        menus = self.get_menus()
        for menu in menus:
            if menu["name"] == menu_name:
                print(f"✓ Found menu by name: {menu['name']} (ID: {menu['id']})")
                return menu["id"]
        print(f"✗ Menu '{menu_name}' not found", file=sys.stderr)
        return None

    def get_primary_menu_id(self, menu_name: Optional[str] = None) -> Optional[int]:
        """Get the primary menu ID."""
        menus = self.get_menus()
        if not menus:
            print("✗ No menus found in WordPress", file=sys.stderr)
            return None

        if menu_name:
            for menu in menus:
                if menu["name"] == menu_name:
                    print(f"✓ Found menu by name: {menu['name']} (ID: {menu['id']})")
                    return menu["id"]
            print(f"✗ Menu '{menu_name}' not found. Available menus:", file=sys.stderr)
            for menu in menus:
                print(f"  - {menu['name']}", file=sys.stderr)
            return None

        for menu in menus:
            if menu["slug"] in ("primary", "main", "header"):
                print(f"✓ Found primary menu: {menu['name']} (ID: {menu['id']})")
                return menu["id"]

        first = menus[0]
        print(f"✓ Using first available menu: {first['name']} (ID: {first['id']})")
        return first["id"]

    def _get_all_menu_items(self, menu_name_or_id) -> list:
        """Get all items in a menu (by name or ID)."""
        # WP-CLI menu item list needs the menu slug/name, not the term_id.
        # Resolve the ID to a slug if necessary.
        menus = self.get_menus()
        menu_slug = None
        for m in menus:
            if m["id"] == menu_name_or_id or m["name"] == str(menu_name_or_id) or m["slug"] == str(menu_name_or_id):
                menu_slug = m["slug"]
                break
        if menu_slug is None:
            menu_slug = str(menu_name_or_id)

        result = self._wp("menu", "item", "list", menu_slug, "--format=json")
        if result.returncode != 0:
            return []
        try:
            return json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            return []

    def add_custom_menu_item(self, menu_id: int, title: str, url: str,
                             parent_id: int = 0, order: int = 0) -> Optional[dict]:
        """Add a custom link to a WordPress menu."""
        menus = self.get_menus()
        menu_slug = None
        for m in menus:
            if m["id"] == menu_id:
                menu_slug = m["slug"]
                break
        if menu_slug is None:
            print(f"✗ Menu ID {menu_id} not found", file=sys.stderr)
            return None

        cmd = ["menu", "item", "add-custom", menu_slug, title, url, f"--position={order}"]
        if parent_id > 0:
            cmd.append(f"--parent-id={parent_id}")
        cmd.append("--porcelain")  # return just the new item ID

        result = self._wp(*cmd)
        if result.returncode == 0:
            item_id = result.stdout.strip()
            print(f"✓ Added menu item: {title} (ID: {item_id})")
            return {"id": int(item_id) if item_id.isdigit() else 0, "title": title, "url": url}
        else:
            print(f"✗ Failed to add menu item {title}: {result.stderr.strip()}", file=sys.stderr)
            return None

    def get_menu_items_by_title(self, menu_id: int, title: str) -> list:
        """Get menu items matching a title."""
        import html as html_mod
        items = self._get_all_menu_items(menu_id)
        normalized_title = html_mod.unescape(title).strip().lower()
        matching = []
        for item in items:
            item_title = item.get("title", "")
            if html_mod.unescape(item_title).strip().lower() == normalized_title:
                matching.append({
                    "id": int(item.get("db_id", item.get("ID", 0))),
                    "title": {"rendered": item_title},
                    "url": item.get("link", item.get("url", "")),
                })
        return matching

    def get_menu_items_by_url(self, menu_id: int, url: str) -> list:
        """Get menu items matching a URL."""
        items = self._get_all_menu_items(menu_id)
        normalized_url = url.strip().lower().rstrip("/")
        matching = []
        for item in items:
            item_url = item.get("link", item.get("url", ""))
            if item_url.strip().lower().rstrip("/") == normalized_url:
                matching.append({
                    "id": int(item.get("db_id", item.get("ID", 0))),
                    "title": {"rendered": item.get("title", "")},
                    "url": item_url,
                })
        return matching

    # ------------------------------------------------------------------
    # Page helpers
    # ------------------------------------------------------------------

    def get_page_by_slug(self, slug: str) -> Optional[dict]:
        """Get WordPress page by slug."""
        data = self._wp_json("post", "list", "--post_type=page",
                             f"--name={slug}", "--posts_per_page=1",
                             "--fields=ID,post_title,post_name")
        if data and len(data) > 0:
            page = data[0]
            return {"id": int(page["ID"]), "title": page.get("post_title", ""), "slug": page.get("post_name", "")}
        return None

    def create_page(self, title: str, slug: str, content: str, parent_id: int = 0) -> bool:
        """Create a new WordPress page."""
        cmd = [
            "post", "create", "--post_type=page",
            f"--post_title={title}",
            f"--post_name={slug}",
            "--post_status=publish",
            "--post_content=-",  # read from stdin
            "--porcelain",
        ]
        if parent_id > 0:
            cmd.append(f"--post_parent={parent_id}")

        # WP-CLI reads content from stdin when --post_content=-
        # Actually, WP-CLI doesn't support --post_content=- ; pass content directly
        cmd.remove("--post_content=-")
        cmd.append(f"--post_content={content}")

        result = self._wp(*cmd)
        if result.returncode == 0:
            page_id = result.stdout.strip()
            print(f"✓ Created WordPress page: {title} (ID: {page_id}, slug: {slug})")
            return True
        else:
            print(f"✗ Failed to create page {title}: {result.stderr.strip()}", file=sys.stderr)
            return False

    def update_page(self, page_id: int, title: str, content: str) -> bool:
        """Update an existing WordPress page."""
        result = self._wp(
            "post", "update", str(page_id),
            f"--post_title={title}",
            f"--post_content={content}",
        )
        if result.returncode == 0:
            print(f"✓ Updated WordPress page: {title} (ID: {page_id})")
            return True
        else:
            print(f"✗ Failed to update page {page_id}: {result.stderr.strip()}", file=sys.stderr)
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
    """Test WordPress sync functionality via WP-CLI"""

    domain = "sensemagic.nl"

    wp_sync = WordPressSync(domain)
    success = wp_sync.test_connection()

    if not success:
        print("\n" + "="*70, file=sys.stderr)
        print("ERROR: WP-CLI connection failed!", file=sys.stderr)
        print("Check that PHP and WP-CLI are available:", file=sys.stderr)
        print(f"  PHP:    {WordPressSync.PHP_BIN}", file=sys.stderr)
        print(f"  WP-CLI: {WordPressSync.WP_CLI}", file=sys.stderr)
        print(f"  WP dir: {WordPressSync.WP_PATH}", file=sys.stderr)
        print("="*70 + "\n", file=sys.stderr)
        sys.exit(1)

    # Test menu functionality
    print("\nTesting menu functionality:")
    print("-" * 60)

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

