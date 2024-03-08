#!/bin/sh

email=""
certificate_name="ipnos-linux"

create_git_certificate() {
    # 
    certificate_path="$HOME/.ssh/${certificate_name}"
    echo "${certificate_path}"

    if [ -f ${certificate_path} ]; then
	echo "key name already exist. change name"
	return 1
    else
	echo "key name doesn't exist. creating.."

	# --- generating key and adding to ssh-agent
	ssh-keygen -t rsa -b 4096 -C "${email}" -f ${certificate_path}
	ssh-add ${certificate_path}
    fi

}

create_git_certificate

