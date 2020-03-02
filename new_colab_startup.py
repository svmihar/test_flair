import subprocess as sp
import os
cmd = ['apt-get update && apt-get upgrade -y', 'apt install zsh tmux vim git wget -y',
        "git config --global user.email 'svmihar@gmail.com'", "git config --global user.name svmihar"]

for c in cmd:
    sp.run(c, shell=True)

os.system('sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh) -y"')

print('\n\n\n\n\nfinished initializing')
