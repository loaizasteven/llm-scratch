#!/usr/bin/python

REPO="venv"
REPONAME="llm-scratch"
DIRECTORY="llm-builder-env"
REQFILE="$HOME/git/$REPONAME/requirements.txt"

function activate () {
  source $HOME/git/$REPO/$DIRECTORY/bin/activate
  # logging info
  echo "Currently activated $VIRTUAL_ENV"
  python --version
  # pip installation
  pip install --upgrade pip
  if [ -d $REQFILE ]; then
    echo "pip installation in verbose mode..."
    pip install -r $REQFILE --upgrade-strategy only-if-needed
  fi
}

if [ -d "$HOME/git/$REPO/$DIRECTORY" ]; then
    activate
else
    echo "Creating venv"
    /usr/local/bin/python3 -m venv "$HOME/git/$REPO/$DIRECTORY"

    activate
fi