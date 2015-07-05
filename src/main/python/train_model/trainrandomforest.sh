#!/usr/bin/env bash
rm -rf model/
/path/to/spark/bin/spark-submit --master local[2] --driver-memory 4g trainrandomforest.py
