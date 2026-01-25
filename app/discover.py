from pathlib import Path
import importlib
import subprocess
import traceback
import sys
import shutil
import argparse
from fastapi import APIRouter
from wordpress_sync import WordPressSync

# Import secrets - REQUIRED, no fallback
try:
    import mysecrets
except ImportError:
    print("\n" + "="*70, file=sys.stderr)
    print("ERROR: mysecrets.py file not found!", file=sys.stderr)
    print("="*70, file=sys.stderr)
    print("WordPress sync requires app/mysecrets.py file with credentials.", file=sys.stderr)
    print("\nTo create it:", file=sys.stderr)
    print("  1. cp app/mysecrets.py.example app/mysecrets.py", file=sys.stderr)
    print("  2. Edit app/mysecrets.py and add your WordPress credentials", file=sys.stderr)
    print("  3. Generate application password in WordPress admin", file=sys.stderr)
    print("="*70 + "\n", file=sys.stderr)
    sys.exit(1)

# Get credentials - REQUIRED fields, no defaults
if not hasattr(mysecrets, 'WP_USER'):
    print("ERROR: WP_USER not defined in mysecrets.py", file=sys.stderr)
    sys.exit(1)

if not hasattr(mysecrets, 'WP_APP_PASSWORD'):
    print("ERROR: WP_APP_PASSWORD not defined in mysecrets.py", file=sys.stderr)
    sys.exit(1)

WP_USER = mysecrets.WP_USER
WP_APP_PASSWORD = mysecrets.WP_APP_PASSWORD

# Optional: WordPress connection settings
WP_BASE_URL = getattr(mysecrets, 'WP_BASE_URL', None)
WP_VERIFY_SSL = getattr(mysecrets, 'WP_VERIFY_SSL', False)

# Validate credentials are not empty or placeholder
if not WP_USER or WP_USER == "admin":
    print("ERROR: WP_USER in mysecrets.py must be set to your actual WordPress username", file=sys.stderr)
    sys.exit(1)

if not WP_APP_PASSWORD or WP_APP_PASSWORD.startswith("xxxx"):
    print("ERROR: WP_APP_PASSWORD in mysecrets.py must be set to your actual application password", file=sys.stderr)
    print("Generate one in WordPress admin: Users → Profile → Application Passwords", file=sys.stderr)
    sys.exit(1)

class Discover:
    DOMAIN = "sensemagic.nl"
    APP_PORT = 9000
    ROUTER_DIR = "routers"
    SUPERVISOR_SERVICE = "fastapi"
    GIT_BRANCH = "main"

    # WordPress credentials (loaded from secrets.py file - REQUIRED)
    WP_USER = WP_USER
    WP_APP_PASSWORD = WP_APP_PASSWORD
    WP_BASE_URL = WP_BASE_URL
    WP_VERIFY_SSL = WP_VERIFY_SSL

    def __init__(self):
        self.routers = {}
        self.base_path = Path(__file__).parent
        self.project_root = self.base_path.parent
        self.directives_path = self.base_path / "nginx_directives.txt"
        self.router_path = self.base_path / self.ROUTER_DIR

    def get_routers(self):
        # 1 Get the router files, they must start with app_ and contain a router variable
        #   You can disable a file by adding a global variable ENABLE=False
        package = sys.modules[__name__].__package__
        router_files = self.router_path.glob("app_*.py")
        routers = []
        for s in router_files:
            try:
                if package is None:
                    mod = importlib.import_module(f"{self.ROUTER_DIR}.{s.stem}")
                else:
                    mod = importlib.import_module(f".{s.stem}", f"{package}.{self.ROUTER_DIR}")

                enabled = getattr(mod, "enabled", True)
                router = getattr(mod, "router", None)
                if not enabled:
                    print(f"router {s.stem} is not enabled")
                    continue
                if router is None or not isinstance(router,  APIRouter):
                    print(f"router {s.stem} has no valid router variable", file=sys.stderr)
                    continue
                routers.append((s.stem, router))
                print(f"Added router {s.stem}")
            except ValueError as ex:
                print(f"Could not load module {s}")
                traceback.print_exc(file=sys.stderr)
        routers.sort()
        self.routers = { name:r for name, r in routers}

    def make_nginx_blocks(self):
        nginx_blocks = []

        # Add static files block - PROXY to FastAPI (FastAPI serves files correctly)
        static_block = f"""
# Static files (CSS, JS, images) - proxied to FastAPI
location /static/ {{
    proxy_pass http://127.0.0.1:{self.APP_PORT}/static/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_http_version 1.1;
    proxy_buffering off;
    
    # Cache headers for static files
    expires 1y;
    add_header Cache-Control "public, immutable";
}}
"""
        nginx_blocks.append(static_block)

        # Add proxy blocks for each API route
        for prefix, router in self.routers.items():
            block = f"""
location /{prefix}/ {{
    proxy_pass http://127.0.0.1:{self.APP_PORT}/{prefix}/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_http_version 1.1;
    proxy_buffering off;
}}
"""
            nginx_blocks.append(block)
        #
        with open(self.directives_path,  "w") as f:
            f.write("\n".join(nginx_blocks))

    def apply_nginx(self):
        target = Path(f"/var/www/vhosts/system/{self.DOMAIN}/conf/vhost_nginx.conf")
        print(f"{target} {target.is_file()}")
        shutil.copy(self.directives_path, target)
        subprocess.run(["systemctl", "restart", "nginx"])

    def git_pull(self):
        """Pull the latest code from Git master branch"""
        print(f"Pulling latest code from Git ({self.GIT_BRANCH} branch)...")
        try:
            result = subprocess.run(
                ["git", "pull", "--rebase", "origin", self.GIT_BRANCH],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            print("Git pull completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error pulling from Git: {e}", file=sys.stderr)
            print(f"stdout: {e.stdout}", file=sys.stderr)
            print(f"stderr: {e.stderr}", file=sys.stderr)
            return False

    def restart_supervisor(self):
        """Restart the FastAPI supervisor service"""
        print(f"Restarting supervisor service: {self.SUPERVISOR_SERVICE}...")
        try:
            result = subprocess.run(
                ["supervisorctl", "restart", self.SUPERVISOR_SERVICE],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            print(f"Supervisor service {self.SUPERVISOR_SERVICE} restarted successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error restarting supervisor: {e}", file=sys.stderr)
            print(f"stdout: {e.stdout}", file=sys.stderr)
            print(f"stderr: {e.stderr}", file=sys.stderr)
            return False

    def sync_wordpress(self):
        """Sync discovered routers to WordPress pages"""
        # Credentials are already validated at module import time
        # If we get here, they must be valid
        try:
            # Use configuration from secrets.py
            wp_sync = WordPressSync(
                self.DOMAIN,
                self.WP_USER,
                self.WP_APP_PASSWORD,
                base_url=self.WP_BASE_URL,      # From secrets.py (optional)
                verify_ssl=self.WP_VERIFY_SSL   # From secrets.py (optional)
            )
            return wp_sync.sync_routers_to_wordpress(self.routers, self.APP_PORT)
        except Exception as e:
            print(f"Error syncing to WordPress: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # Re-raise the exception instead of returning False
            # This ensures errors are not hidden
            raise

    def sync_wordpress_menu(self, menu_name=None, articles_parent="Articles", applications_parent="Applications"):
        """
        Sync discovered routers to WordPress menu as sub-menu items

        Parameters:
        menu_name: Optional menu name to sync to (auto-detects if None)
        articles_parent: Title of parent menu item for articles (default: "Articles")
        applications_parent: Title of parent menu item for applications (default: "Applications")
        """
        # Credentials are already validated at module import time
        try:
            # Use configuration from secrets.py
            wp_sync = WordPressSync(
                self.DOMAIN,
                self.WP_USER,
                self.WP_APP_PASSWORD,
                base_url=self.WP_BASE_URL,      # From secrets.py (optional)
                verify_ssl=self.WP_VERIFY_SSL   # From secrets.py (optional)
            )
            return wp_sync.sync_routers_as_submenu(
                self.routers,
                menu_name=menu_name,
                articles_parent_title=articles_parent,
                applications_parent_title=applications_parent
            )
        except Exception as e:
            print(f"Error syncing to WordPress menu: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # Re-raise the exception instead of returning False
            # This ensures errors are not hidden
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Discover FastAPI routers, generate nginx config, and manage deployment"
    )
    parser.add_argument(
        "--git-pull",
        action="store_true",
        help="Pull latest code from Git master branch before processing"
    )
    parser.add_argument(
        "--update-nginx",
        action="store_true",
        help="Update nginx configuration and restart nginx service"
    )
    parser.add_argument(
        "--restart-supervisor",
        action="store_true",
        help="Restart supervisor fastapi service after processing"
    )
    parser.add_argument(
        "--sync-wordpress",
        action="store_true",
        help="Sync discovered routers to WordPress pages (requires WP_APP_PASSWORD env var)"
    )
    parser.add_argument(
        "--sync-wordpress-menu",
        action="store_true",
        help="Sync discovered routers to WordPress menu as sub-menu items"
    )
    parser.add_argument(
        "--menu-name",
        type=str,
        default=None,
        help="WordPress menu name to sync to (auto-detects if not specified)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Perform all actions: git pull, router discovery, nginx config, WordPress sync (pages + menu), and supervisor restart"
    )

    args = parser.parse_args()

    dis = Discover()

    # If --all is specified, enable all actions
    if args.all:
        args.git_pull = True
        args.update_nginx = True
        args.restart_supervisor = True
        args.sync_wordpress = True
        args.sync_wordpress_menu = True

    # Pull from Git if requested
    if args.git_pull:
        if not dis.git_pull():
            print("Warning: Git pull failed, continuing anyway...", file=sys.stderr)

    # Always perform router discovery
    dis.get_routers()
    dis.make_nginx_blocks()

    # Update nginx if requested
    if args.update_nginx:
        dis.apply_nginx()
    else:
        print("Nginx configuration generated but not applied (use --update-nginx to apply)")

    # Sync to WordPress if requested
    if args.sync_wordpress:
        dis.sync_wordpress()

    # Sync to WordPress menu if requested
    if args.sync_wordpress_menu:
        dis.sync_wordpress_menu(menu_name=args.menu_name)

    # Restart supervisor if requested
    if args.restart_supervisor:
        dis.restart_supervisor()

if __name__ == "__main__":
    main()
