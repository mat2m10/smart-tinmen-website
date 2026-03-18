# Development Setup (Windows + WSL2 + Ubuntu + Python)

This guide walks you through setting up a modern Python development
environment on **Windows** using **WSL2**, **Ubuntu**, **VS Code**, and
**pyenv**.

Please follow the steps in order.

------------------------------------------------------------------------

## GitHub Account

1.  Create an account: https://github.com/join\
2.  Upload a profile picture and verify your name:
    https://github.com/settings/profile\
3.  Enable Two-Factor Authentication (2FA):\
    https://docs.github.com/en/authentication/securing-your-account-with-two-factor-authentication-2fa/configuring-two-factor-authentication

------------------------------------------------------------------------

## Windows Version

You need **Windows 10 (version 2004+)** or **Windows 11**.

Check your version: - Press `Windows + R` - Type `winver` - Press
`Enter`

If Windows 10 is below 2004, update via Windows Update: - Press
`Windows + R` - Type `ms-settings:windowsupdate` - Press `Enter` - Click
**Check updates**

------------------------------------------------------------------------

## Virtualization

Check if virtualization is enabled: - Press `Windows + R` - Type
`taskmgr` - Press `Enter` - Go to **Performance → CPU**

You should see: **Virtualization: Enabled**

If not enabled, activate it in BIOS/UEFI (Intel VT-x / AMD-V / SVM).

------------------------------------------------------------------------

## Obvious but lets say it anyways

Be connected to a microsoft account
Make sure that windows is activated
------------------------------------------------------------------------

## Install Windows Terminal (optional but recommended)

Open Command Prompt and run:
``` powershell
winget install Microsoft.WindowsTerminal --source winget
```

------------------------------------------------------------------------
## Install WSL2 + Ubuntu

Open **Command Prompt as Administrator**: - Press `Windows + R` - Type
`cmd` - Press `Ctrl + Shift + Enter`

Run:

``` powershell
wsl --install
```

Restart your computer when finished.

------------------------------------------------------------------------

## Ubuntu Setup

### First launch

Ubuntu will ask for: - a **username** (lowercase, one word) - a
**password** (typing won't show characters --- this is normal)

### Check WSL version

``` bash
wsl -l -v
```

If Ubuntu shows version `1`, convert it:

``` bash
wsl --set-version Ubuntu 2
```

### Check your Linux username

``` bash
whoami
```

If it prints `root`, stop and fix your Ubuntu user setup.

------------------------------------------------------------------------

## Install VS Code

Download: https://code.visualstudio.com/download

### Connect VS Code to WSL

From Ubuntu:

``` bash
code --install-extension ms-vscode-remote.remote-wsl
code .
```

You should see `WSL: Ubuntu` in the bottom-left corner.

------------------------------------------------------------------------

## Install Essential Tools

``` bash
sudo apt update
sudo apt install -y curl git imagemagick jq unzip vim zsh tree
```

------------------------------------------------------------------------

## GitHub CLI (optional)

Install GitHub CLI (`gh`):

```bash
sudo apt remove -y gitsome # gh command can conflict with gitsome if installed
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install -y gh
```

Check:

```bash
gh --version
```

Login (SSH recommended):

```bash
gh auth login -s 'user:email' -w --git-protocol ssh
```

Verify:

```bash
gh auth status
```
------------------------------------------------------------------------

## Install Oh My Zsh (optional)

``` bash
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

------------------------------------------------------------------------

## Install pyenv

``` bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
exec zsh
```

Install build dependencies:

``` bash
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev sqlite3 libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev python3-dev
```

## Configure .zshrc

Open it in VS Code:

``` bash
cd
code .zshrc
```

Replace its contents with:

``` zsh
ZSH=$HOME/.oh-my-zsh

# You can change the theme with another one from https://github.com/robbyrussell/oh-my-zsh/wiki/themes
ZSH_THEME="robbyrussell"

# Useful oh-my-zsh plugins for Le Wagon bootcamps
plugins=(git gitfast last-working-dir common-aliases history-substring-search ssh-agent)

# (macOS-only) Prevent Homebrew from reporting - https://github.com/Homebrew/brew/blob/master/docs/Analytics.md
export HOMEBREW_NO_ANALYTICS=1

# Disable warning about insecure completion-dependent directories
ZSH_DISABLE_COMPFIX=true

# Actually load Oh-My-Zsh
source "${ZSH}/oh-my-zsh.sh"
unalias rm # No interactive rm by default (brought by plugins/common-aliases)
unalias lt # we need `lt` for https://github.com/localtunnel/localtunnel


# Load pyenv (to manage your Python versions)
export PYENV_VIRTUALENV_DISABLE_PROMPT=1

# Load nvm (to manage your node versions)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# Call `nvm use` automatically in a directory with a `.nvmrc` file
autoload -U add-zsh-hook
load-nvmrc() {
  if nvm -v &> /dev/null; then
    local node_version="$(nvm version)"
    local nvmrc_path="$(nvm_find_nvmrc)"

    if [ -n "$nvmrc_path" ]; then
      local nvmrc_node_version=$(nvm version "$(cat "${nvmrc_path}")")

      if [ "$nvmrc_node_version" = "N/A" ]; then
        nvm install
      elif [ "$nvmrc_node_version" != "$node_version" ]; then
        nvm use --silent
      fi
    elif [ "$node_version" != "$(nvm version default)" ]; then
      nvm use default --silent
    fi
  fi
}
type -a nvm > /dev/null && add-zsh-hook chpwd load-nvmrc
type -a nvm > /dev/null && load-nvmrc

# Rails and Ruby uses the local `bin` folder to store binstubs.
# So instead of running `bin/rails` like the doc says, just run `rails`
# Same for `./node_modules/.bin` and nodejs
export PATH="./bin:./node_modules/.bin:${PATH}:/usr/local/sbin"

# Store your own aliases in the ~/.aliases file and load the here.
[[ -f "$HOME/.aliases" ]] && source "$HOME/.aliases"

# Encoding stuff for the terminal
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

export BUNDLER_EDITOR=code
export EDITOR=code

# Set ipdb as the default Python debugger
export PYTHONBREAKPOINT=ipdb.set_trace
export BUNDLER_EDITOR="subl $@ >/dev/null 2>&1"

# Load pyenv automatically by adding
# the following to ~/.zshrc:

export PATH="$HOME/.pyenv/bin:$PATH"


eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
type -a pyenv > /dev/null && eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init - 2> /dev/null)" && RPROMPT+='[🐍 $(pyenv version-name)]'

export BROWSER=google-chrome

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
```

Reload:

``` bash
source ~/.zshrc
```

------------------------------------------------------------------------

## Add your github username
``` zsh
export COURSE_USERNAME="your_github_username"
```

Install Python:

``` bash
pyenv install 3.12.9
pyenv global 3.12.9
exec zsh
```

Verify:

``` bash
python --version
```

------------------------------------------------------------------------

## Install pyenv-virtualenv

``` bash
git clone https://github.com/pyenv/pyenv-virtualenv.git "$(pyenv root)/plugins/pyenv-virtualenv"
exec zsh
```

------------------------------------------------------------------------


## Create Virtual Environment

``` bash
pyenv virtualenv 3.12.9 myenv
pyenv global myenv
```

Upgrade pip:

``` bash
pip install --upgrade pip
```

Install example packages:

``` bash
pip install numpy pandas scikit-learn jupyter
```

------------------------------------------------------------------------

## Run Jupyter

``` bash
jupyter notebook
```

Stop with `Ctrl + C`.

------------------------------------------------------------------------

## install node.js to check the slides

``` bash
sudo apt update
sudo apt install nodejs npm
```
## Clone the course repo into your home folder and install pytests

Make a folder named after your GitHub username, then clone the course repo inside it.

```bash
cd ~
pip install pytest
gh repo clone smart-tinmen/python_class
cd python_class
```

Setup complete 🚀