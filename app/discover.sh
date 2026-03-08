#!/bin/bash
# Activate the virtual environment
source /home/venv/fapi/bin/activate

cd /home/projects
chown projects * -R

# Move into the app directory
cd /home/projects/sensemagic/app

# --- Pre-flight: verify local WordPress REST API is reachable ---
# Apache on 127.0.0.1:7080 must NOT redirect /wp-json/ to HTTPS.
# If this check fails, the Plesk vhost.conf fix is missing or was overwritten.
WP_CHECK=$(curl -s -o /dev/null -w "%{http_code}" -H "Host: sensemagic.nl" \
    "http://127.0.0.1:7080/wp-json/wp/v2/" 2>/dev/null)

if [ "$WP_CHECK" = "301" ] || [ "$WP_CHECK" = "302" ]; then
    echo "========================================================================"
    echo "ERROR: WordPress REST API on 127.0.0.1:7080 returns a $WP_CHECK redirect."
    echo ""
    echo "Apache is redirecting local /wp-json/ requests to HTTPS, which is"
    echo "unreachable from localhost. This happens when the Plesk vhost.conf"
    echo "fix is missing or was overwritten (e.g. after a certificate renewal)."
    echo ""
    echo "FIX: Re-install the local API bypass rule:"
    echo ""
    echo "  cp /home/projects/sensemagic/app/plesk_apache_local_api.conf \\"
    echo "     /var/www/vhosts/system/sensemagic.nl/conf/vhost.conf"
    echo "  /usr/sbin/plesk repair web sensemagic.nl"
    echo ""
    echo "Then re-run this script."
    echo "========================================================================"
    exit 1
elif [ "$WP_CHECK" = "000" ]; then
    echo "WARNING: Could not reach Apache on 127.0.0.1:7080 (curl returned 000)."
    echo "Apache may not be running. Continuing anyway..."
fi

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


