#!/usr/bin/env bash

source activate molecular-tktd
pytest -m "not slow"