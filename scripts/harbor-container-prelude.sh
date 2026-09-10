#!/bin/bash
# Sourced inside the container, immediately before harbor's own bootstrap.sh.
#
# Everything here repairs a consequence of running the container in an unprivileged user
# namespace (see scripts/singularity-no-fakeroot for why that namespace is necessary at all).
# Each item was diagnosed from an actual failure, not added defensively.

# --- 1. apt drops privileges to the _apt user, which a single-mapping userns forbids ----------
# Symptom: "E: seteuid 42 failed", "E: setgroups 0 failed", then "Unable to locate package
# python3" because the package lists never refreshed. Telling apt to stay root is the standard
# fix and is safe here: the container is disposable and we are already uid 0 inside it.
mkdir -p /etc/apt/apt.conf.d 2>/dev/null
printf 'APT::Sandbox::User "root";\n' > /etc/apt/apt.conf.d/99-harbor-userns 2>/dev/null

# --- 2. dpkg cannot rename into directories that exist only in the image's lower layer --------
# Symptom: unpacking tmux fails with "unable to install new version of './usr/share/doc/tmux':
# Invalid cross-device link". harbor's server sets force-overwrite/force-unsafe-io, but that
# does not help a cross-device rename, and it only runs AFTER bootstrap has already tried to
# install tmux. Excluding documentation avoids touching those directories at all; nothing in a
# benchmark run reads man pages.
mkdir -p /etc/dpkg/dpkg.cfg.d 2>/dev/null
printf 'path-exclude=/usr/share/doc/*\npath-exclude=/usr/share/man/*\npath-exclude=/usr/share/info/*\nforce-unsafe-io\nforce-overwrite\n' \
    > /etc/dpkg/dpkg.cfg.d/99-harbor-userns 2>/dev/null

# --- 3. bootstrap.sh tests the literal path /usr/bin/python3 ----------------------------------
# Task images are Debian with the interpreter at /usr/local/bin/python3, so bootstrap declares
# python missing and tries to install it. Point the path it checks at the python already there.
if [ ! -x /usr/bin/python3 ]; then
    _p=$(command -v python3 2>/dev/null)
    [ -n "$_p" ] && ln -sf "$_p" /usr/bin/python3 2>/dev/null
    unset _p
fi

# --- 4. `su` cannot work in this namespace ----------------------------------------------------
# harbor runs every task command as `su <user> -s /bin/bash -c <cmd>` (environments/singularity/
# singularity.py). Inside the namespace there is no uid-0 passwd entry, and even after adding
# one `su` fails with "cannot set groups: Operation not permitted", because setgroups is denied
# in a single-mapping user namespace. We are already uid 0, so the privilege change su would
# perform is a no-op -- exactly the reasoning behind harbor's own setup_fake_sudo(). This
# replaces su with a shim that skips the user/shell arguments and runs the command.
# /usr/local/bin precedes /usr/bin on the Debian PATH, which is the same assumption harbor's
# fake sudo already relies on.
# Written to BOTH paths: harbor's server invokes a bare `su`, and the PATH its exec environment
# uses is not guaranteed to put /usr/local/bin first, so /usr/bin/su is overridden as well.
mkdir -p /usr/local/bin 2>/dev/null
cat > /usr/local/bin/su <<'HARBOR_SU'
#!/bin/bash
# Stand-in for su inside an unprivileged user namespace where the process is already uid 0.
# Understands the form harbor emits: su <user> [-s <shell>] -c <command>
shell=/bin/bash
while [ $# -gt 0 ]; do
    case "$1" in
        -c) shift; exec "$shell" -c "$1" ;;
        -s) shift; [ -n "${1:-}" ] && shell="$1" ;;
        -|-l|--login|-m|-p|--preserve-environment) ;;
        --) shift; break ;;
        *)  ;;                      # the target user: ignored, we are already root
    esac
    shift
done
exec "$shell"
HARBOR_SU
chmod 755 /usr/local/bin/su 2>/dev/null
cp -f /usr/local/bin/su /usr/bin/su 2>/dev/null && chmod 755 /usr/bin/su 2>/dev/null

# Also give the namespace a root identity, so anything that looks the uid up by name works.
grep -q '^root:' /etc/passwd 2>/dev/null || \
    echo 'root:x:0:0:root:/root:/bin/bash' >> /etc/passwd 2>/dev/null
grep -q '^root:' /etc/group 2>/dev/null || echo 'root:x:0:' >> /etc/group 2>/dev/null

# --- 5. tmux, which dpkg cannot install onto this overlay -------------------------------------
# The terminal agents drive a tmux session, and the task images do not ship tmux. Installing it
# with apt fails even with the dpkg options above:
#   dpkg: error processing archive .../tmux_3.5a-3_amd64.deb (--unpack):
#    unable to install new version of './usr/share/doc/tmux': Invalid cross-device link
# dpkg unpacks by renaming into place, and a rename onto singularity's overlay returns EXDEV.
# path-exclude does not avoid it either, because the directory itself is still renamed. So the
# packages are downloaded and extracted directly with dpkg-deb -x, which just writes files and
# never renames. tmux is not registered in the dpkg database this way, which does not matter --
# nothing here queries it, and the container is discarded after the trial.
if ! command -v tmux > /dev/null 2>&1 && command -v apt-get > /dev/null 2>&1; then
    apt-get update -qq > /dev/null 2>&1
    # --download-only puts tmux AND its dependencies in the archive cache without unpacking.
    apt-get install -y --download-only tmux > /dev/null 2>&1
    for _deb in /var/cache/apt/archives/*.deb; do
        [ -f "$_deb" ] && dpkg-deb -x "$_deb" / 2>/dev/null
    done
    unset _deb
    ldconfig 2>/dev/null || true
fi

# One line in the trial log, so a future failure can be told apart from the prelude not running
# at all. Note the caller must NOT redirect this away -- an earlier version sourced the prelude
# with 2>/dev/null and spent a debugging round unable to tell the two cases apart.
echo "[harbor-prelude] applied: python3=$(command -v python3 2>/dev/null) su=$(command -v su 2>/dev/null) tmux=$(command -v tmux 2>/dev/null) uid=$(id -u)" >&2

true    # never let a failed repair abort the bootstrap
