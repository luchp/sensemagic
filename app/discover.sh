#!/bin/bash
# Activate the virtual environment
source /home/venv/fapi/bin/activate

cd /home/projects
chown projects * -R

# Move into the app directory
cd /home/projects/sensemagic/app

# Do a discovery, pass all arguments to discover.py
# Usage examples:
#   ./discover.sh                              # Just discover routers (no nginx update)
#   ./discover.sh --update-nginx               # Discover routers and update nginx
#   ./discover.sh --git-pull --update-nginx    # Pull from Git, discover, and update nginx
#   ./discover.sh --restart-supervisor         # Discover and restart supervisor
#   ./discover.sh --sync-wordpress             # Sync routers to WordPress pages
#   ./discover.sh --sync-wordpress-menu        # Sync routers to WordPress menu items
#   ./discover.sh --sync-wordpress --sync-wordpress-menu  # Sync both pages and menu
#   ./discover.sh --menu-name "Main Menu"      # Specify WordPress menu by name
#   ./discover.sh --all                        # Do everything: git pull, discover, update nginx, WordPress sync (pages + menu), restart supervisor
python discover.py "$@"


