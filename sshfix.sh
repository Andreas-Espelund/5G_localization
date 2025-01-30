#!/bin/bash

eval "$(ssh-agent -s)"

ssh-add ~/.ssh/hpc

ssh -T git@github.com