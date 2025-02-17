diff_HEAD_prev_commit() {
    git diff HEAD^^..HEAD $1
}

diff_HEAD_prev_commit main.c


# see all git commits/merge in one line
git log --oneline --graph --all --decorate

#
