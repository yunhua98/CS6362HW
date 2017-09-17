#!/bin/bash

java -cp commons-cli-1.2.jar:commons-math3-3.2.jar:. cs362.Learn -mode test -model_file $1 -data $2 -predictions_file $3 -task $4
