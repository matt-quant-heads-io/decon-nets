# Sets the env var PROJECT_ROOT to the directory of this file.
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [[ $0 == $BASH_SOURCE ]]; then 
        echo "source this script to set the PROJECT_ROOT env var."     
else
        echo "PROJECT_ROOT is now $PROJECT_ROOT"
fi