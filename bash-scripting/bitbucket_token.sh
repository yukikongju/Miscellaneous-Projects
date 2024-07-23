#!/bin/sh

email=""
certificate_name="ipnos-linux"
certificate_path="$HOME/.ssh/${certificate_name}"

create_git_certificate() {
    # 
    echo "${certificate_path}"

    if [ -f ${certificate_path} ]; then
	echo "key name already exist. change name"
	return 1
    else
	echo "key name doesn't exist. creating.."

	# --- generating key and adding to ssh-agent
	ssh-keygen -t rsa -b 4096 -C "${email}" -f ${certificate_path}
	eval "$(ssh-agent)" # start ssh-agent
	ssh-add ${certificate_path}

	echo "${email}"
    fi

}

create_git_certificate

cat ${certificate_path}.pub | pbcopy

