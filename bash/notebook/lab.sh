#!/bin/bash

unset XDG_RUNTIME_DIR
jupyter lab --ip=$(hostname -f) --port=8008 --no-browser --allow-root --ServerApp.token='' --ServerApp.password=''
