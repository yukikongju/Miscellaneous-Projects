diff_HEAD_prev_commit() {
    git diff HEAD^^..HEAD $1
}

diff_HEAD_prev_commit main.c
