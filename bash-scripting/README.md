# Bash Scripting

File management utilities using bash

- [X] Renaming files extension


**Git/Bitbucket Token**

- On *Bitbucket*: use `bitbucket_token.sh` to generate new token and add `<key>.pub` to bitbucket profile
    * Settings > Personal Bitbucket Settings > SSH Keys > Add keys
    * [Configure SSH Key on Bitbucket](https://support.atlassian.com/bitbucket-cloud/docs/configure-ssh-and-two-step-verification/)
- On *Github*: generate token from github [here](https://github.com/settings/tokens) and use the generate token to login. The generated token will be the password
- When pushing from one profile to the other, we have to make sure that the user 
  that was logged in correspond to the actual profile we want. We have to 
  change our user profile using `git_profile.sh`
    * to check current profile `git config --global --list`

