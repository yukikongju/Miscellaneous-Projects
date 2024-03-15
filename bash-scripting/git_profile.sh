#!/bin/sh

github_profile() {
    git config --global user.name "yukikongju"
    git config --global user.email "emulie.chhor@hotmail.com"
}

bitbucket_profile() {
    git config --global user.name "emulie.chhor"
    git config --global user.email "emulie@ipnos.com"
}

view_current_profile(){
    git config --global --list
}

github_profile
view_current_profile
