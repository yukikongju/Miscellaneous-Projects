# Find directory size
du -sh -- *  | sort -rh  # Files and directories, or
du -sh -- */ | sort -rh  # Directories only

# Sort Files in directory by last modified date
ls -ltR --time-style=+"%Y-%m-%d %T" ./ | grep -v '^d' | sort -k6,7 | cut -d' ' -f6-'
